import discord
from discord.ext import commands
import asyncio
import io
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from database import get_db, SessionLocal
from models import User, Report, ReportChunk, Message, MessageRole
from pdf_processor import PDFProcessor
from openai_client import OpenAIClient
from config import DISCORD_TOKEN, BOT_MENTION_NAME, MAX_CHUNKS_PER_QUERY

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents)
pdf_processor = PDFProcessor()
openai_client = OpenAIClient()

class BiomeBot:
    def __init__(self):
        self.processing_users = set()  # Track users currently processing PDFs
    
    async def ensure_user_exists(self, discord_user, db: Session) -> User:
        """Ensure user exists in database"""
        user = db.query(User).filter(User.id == discord_user.id).first()
        if not user:
            user = User(
                id=discord_user.id,
                username=discord_user.display_name or discord_user.name
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    
    async def get_thread_conversation_history(self, report_id: int, db: Session) -> List[Dict[str, str]]:
        """Get full conversation history for a thread/report"""
        messages = db.query(Message).filter(
            Message.report_id == report_id
        ).order_by(Message.id).all()
        
        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return history
    
    async def find_relevant_chunks(self, query: str, report_id: int, db: Session) -> List[str]:
        """Find relevant chunks for query using vector similarity"""
        try:
            # Get query embedding
            query_embedding = openai_client.get_embedding(query)
            
            # Try pgvector similarity search first
            try:
                from pgvector.sqlalchemy import Vector
                
                chunks = db.query(ReportChunk).filter(
                    ReportChunk.report_id == report_id
                ).order_by(
                    ReportChunk.embedding.cosine_distance(query_embedding)
                ).limit(MAX_CHUNKS_PER_QUERY).all()
                
                return [chunk.content for chunk in chunks]
                
            except (ImportError, Exception) as e:
                print(f"pgvector search failed, using fallback: {e}")
                # Fallback: return first few chunks (not ideal but functional)
                chunks = db.query(ReportChunk).filter(
                    ReportChunk.report_id == report_id
                ).order_by(ReportChunk.chunk_idx).limit(MAX_CHUNKS_PER_QUERY).all()
                
                return [chunk.content for chunk in chunks]
                
        except Exception as e:
            print(f"Error finding relevant chunks: {e}")
            return []
    
    async def save_message(self, message_id: int, report_id: int, user_id: Optional[int], 
                          role: str, content: str, db: Session, 
                          input_tokens: int = 0, output_tokens: int = 0, 
                          cost_usd: float = 0.0, chunk_ids: List[int] = None) -> Message:
        """Save message to database"""
        message_record = Message(
            id=message_id,
            report_id=report_id,
            user_id=user_id,
            role=role,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            retrieved_chunk_ids=chunk_ids or []
        )
        db.add(message_record)
        db.commit()
        db.refresh(message_record)
        return message_record
    
    async def process_pdf_upload(self, message: discord.Message, attachment: discord.Attachment, db: Session):
        """Process PDF upload and create thread"""
        user = await self.ensure_user_exists(message.author, db)
        
        try:
            # Download PDF first
            pdf_bytes = await attachment.read()
            
            # Create unique thread name with counter if needed
            base_name = f"🧬 {attachment.filename} - {message.author.display_name}"
            thread_name = base_name
            counter = 1
            
            # Initialize thread variable
            thread = None
            try:
                thread = await message.create_thread(
                    name=thread_name,
                    auto_archive_duration=10080  # 7 days
                )
            except discord.HTTPException as e:
                if e.code == 160004:  # Thread already exists for this message
                    # Create a new thread with incremented name
                    while counter <= 10:  # Limit attempts
                        try:
                            thread_name = f"{base_name} #{counter}"
                            # Create new message to attach thread to
                            temp_msg = await message.channel.send(f"📊 Processing {attachment.filename} (Upload #{counter})")
                            thread = await temp_msg.create_thread(
                                name=thread_name,
                                auto_archive_duration=10080
                            )
                            break
                        except discord.HTTPException:
                            counter += 1
                    
                    if not thread:
                        await message.reply("❌ Unable to create thread after multiple attempts. Please try again.")
                        return
                else:
                    raise e
            
            # Process PDF
            await thread.send("📊 Analyzing your microbiome report...")
            
            processed_data = pdf_processor.process_pdf(pdf_bytes)
            
            # Create report record
            sample_date = None
            if processed_data['metadata'].get('sample_date'):
                # Convert ISO string back to datetime for database field
                from datetime import datetime
                sample_date = datetime.fromisoformat(processed_data['metadata']['sample_date'])
            
            report = Report(
                user_id=user.id,
                thread_id=thread.id,
                original_filename=attachment.filename,
                sample_date=sample_date,
                report_metadata=processed_data['metadata']
            )
            db.add(report)
            db.commit()
            db.refresh(report)
            
            # Create chunks and embeddings
            await thread.send("🔬 Creating knowledge base from your report...")
            
            for chunk_data in processed_data['chunks']:
                try:
                    # Get embedding
                    embedding = openai_client.get_embedding(chunk_data['content'])
                    
                    chunk = ReportChunk(
                        report_id=report.id,
                        chunk_idx=chunk_data['chunk_idx'],
                        content=chunk_data['content']
                    )
                    
                    # Set embedding based on available method
                    try:
                        chunk.embedding = embedding
                    except AttributeError:
                        # Fallback to array storage
                        chunk.embedding_array = embedding
                    
                    db.add(chunk)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_data['chunk_idx']}: {e}")
                    continue
            
            db.commit()
            
            # Initial analysis based on extracted date
            if processed_data['metadata'].get('sample_date'):
                sample_date_str = processed_data['metadata']['sample_date']
                age_months = processed_data['metadata'].get('sample_age_months', 0)
                
                # Convert ISO string to datetime for formatting
                from datetime import datetime
                sample_date_obj = datetime.fromisoformat(sample_date_str)
                
                date_response = f"📅 I see your microbiome report was generated on **{sample_date_obj.strftime('%B %d, %Y')}**\n"
                date_response += f"That's roughly **{age_months} months** ago. Gut profiles can shift fast, so I'll keep that in mind.\n\n"
            else:
                date_response = "Looks like the report date is missing.\nWhen did you take this test? (Month & year is enough.)\n\n"
            
            date_response += "Did you take any antibiotics around the time of the test?"
            
            # Send initial response
            bot_message = await thread.send(date_response)
            
            # Save messages to database
            await self.save_message(
                message_id=message.id,
                report_id=report.id,
                user_id=user.id,
                role=MessageRole.USER.value,
                content=f"[PDF Upload: {attachment.filename}]",
                db=db
            )
            
            await self.save_message(
                message_id=bot_message.id,
                report_id=report.id,
                user_id=None,
                role=MessageRole.BOT.value,
                content=date_response,
                db=db
            )
            
            print(f"✅ Processed PDF for user {user.username}: {report.id}")
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            await message.reply(f"❌ Sorry, I couldn't process your PDF: {str(e)}")
    
    async def handle_thread_message(self, message: discord.Message, db: Session):
        """Handle message in an existing thread"""
        # Find report by thread ID
        report = db.query(Report).filter(Report.thread_id == message.channel.id).first()
        
        if not report:
            return  # Not a report thread
        
        user = await self.ensure_user_exists(message.author, db)
        
        # Save user message
        await self.save_message(
            message_id=message.id,
            report_id=report.id,
            user_id=user.id,
            role=MessageRole.USER.value,
            content=message.content,
            db=db
        )
        
        # Get conversation history
        conversation_history = await self.get_thread_conversation_history(report.id, db)
        
        # Find relevant chunks from the report
        relevant_chunks = await self.find_relevant_chunks(message.content, report.id, db)
        
        # Generate response
        try:
            async with message.channel.typing():
                response_data = openai_client.create_microbiome_analysis(
                    conversation_history=conversation_history,
                    relevant_chunks=relevant_chunks,
                    user_question=message.content
                )
            
            # Send response in chunks if needed (Discord limit: 2000 chars)
            content = response_data['content']
            if len(content) <= 2000:
                bot_message = await message.reply(content)
            else:
                # Split into chunks
                chunks = []
                current_chunk = ""
                
                # Split by paragraphs first
                paragraphs = content.split('\n\n')
                
                for paragraph in paragraphs:
                    # If adding this paragraph would exceed limit, send current chunk
                    if len(current_chunk) + len(paragraph) + 2 > 1950:  # Leave buffer
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                    
                    # If single paragraph is too long, split by sentences
                    if len(paragraph) > 1950:
                        sentences = paragraph.split('. ')
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 2 > 1950:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                    current_chunk = ""
                            current_chunk += sentence + ". "
                    else:
                        current_chunk += paragraph + "\n\n"
                
                # Add remaining content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Send first chunk as reply, rest as follow-ups
                bot_message = await message.reply(chunks[0] if chunks else "Response too long to display.")
                
                # Send remaining chunks
                for chunk in chunks[1:]:
                    await message.channel.send(chunk)
            
            # Save bot message with cost tracking
            chunk_ids = []  # Would need to track which chunks were used
            await self.save_message(
                message_id=bot_message.id,
                report_id=report.id,
                user_id=None,
                role=MessageRole.BOT.value,
                content=response_data['content'],
                db=db,
                input_tokens=response_data['input_tokens'],
                output_tokens=response_data['output_tokens'],
                cost_usd=response_data['cost_usd'],
                chunk_ids=chunk_ids
            )
            
            # Check if we need to automatically send follow-up messages
            await self.check_and_send_followups(message, report, conversation_history, relevant_chunks, db)
            
            print(f"💬 Responded to user {user.username} in report {report.id}")
            
        except Exception as e:
            print(f"Error generating response: {e}")
            await message.reply("❌ Sorry, I encountered an error processing your question. Please try again.")

    async def check_and_send_followups(self, message: discord.Message, report: Report, conversation_history: List[Dict[str, str]], relevant_chunks: List[str], db: Session):
        """Check conversation stage and send automatic follow-up messages"""
        # Analyze conversation to determine stage
        recent_messages = conversation_history[-8:]  # Look at last 8 messages
        
        # Check if user just confirmed/corrected their diet and we need to send executive summary + recommendations
        if len(recent_messages) >= 4:
            user_messages = [msg for msg in recent_messages if msg['role'] == 'user']
            bot_messages = [msg for msg in recent_messages if msg['role'] == 'bot']
            
            # Check if we just provided an executive summary after diet confirmation
            if len(bot_messages) >= 1:
                last_bot_message = bot_messages[-1]['content'].lower()
                
                # Look for executive summary and check if we haven't sent follow-ups yet
                if ('executive summary' in last_bot_message and 
                    'actionable insight' not in last_bot_message and 
                    'feel free to ask' not in last_bot_message):
                    
                    print(f"🎯 Detected executive summary, sending follow-ups...")
                    
                    try:
                        # Send one actionable insight
                        print(f"📝 Sending actionable insight...")
                        await self.send_actionable_insight(message, report, conversation_history, relevant_chunks, db)
                        print(f"✅ Actionable insight sent")
                        
                        # Wait a moment then send Q&A invitation
                        import asyncio
                        await asyncio.sleep(2)
                        print(f"❓ Sending Q&A invitation...")
                        await self.send_qa_invitation(message, report, db)
                        print(f"✅ Q&A invitation sent")
                        
                    except Exception as e:
                        print(f"❌ Error in follow-up messages: {e}")
                        import traceback
                        traceback.print_exc()

    async def send_recommendations(self, message: discord.Message, report: Report, conversation_history: List[Dict[str, str]], relevant_chunks: List[str], db: Session):
        """Send actionable recommendations as a separate message"""
        try:
            # Create prompt specifically for recommendations
            recommendations_prompt = "Based on the conversation history and microbiome data, provide exactly 3 specific, actionable recommendations for improving their microbiome health. Be concise and practical."
            
            response_data = openai_client.create_microbiome_analysis(
                conversation_history=conversation_history,
                relevant_chunks=relevant_chunks,
                user_question=recommendations_prompt
            )
            
            # Ensure the response starts with the correct prefix
            content = response_data['content']
            if not content.lower().startswith('top 3 actionable recommendations'):
                content = f"Top 3 actionable recommendations\n\n{content}"
            
            # Send recommendations message
            rec_message = await message.channel.send(content)
            
            # Save to database
            await self.save_message(
                message_id=rec_message.id,
                report_id=report.id,
                user_id=None,
                role=MessageRole.BOT.value,
                content=content,
                db=db,
                input_tokens=response_data['input_tokens'],
                output_tokens=response_data['output_tokens'],
                cost_usd=response_data['cost_usd']
            )
            
        except Exception as e:
            print(f"Error sending recommendations: {e}")

    async def send_actionable_insight(self, message: discord.Message, report: Report, conversation_history: List[Dict[str, str]], relevant_chunks: List[str], db: Session):
        """Send one actionable insight as a separate message"""
        try:
            # Create prompt for actionable insight
            insight_prompt = "Based on the conversation history and microbiome data, provide exactly one specific, actionable insight the user can implement immediately to improve their gut health. Be concise and practical."
            
            response_data = openai_client.create_microbiome_analysis(
                conversation_history=conversation_history,
                relevant_chunks=relevant_chunks,
                user_question=insight_prompt
            )
            
            # Ensure the response starts with the correct prefix
            content = response_data['content']
            if not content.lower().startswith('one actionable insight'):
                content = f"One actionable insight\n\n{content}"
            
            # Send insight message
            insight_message = await message.channel.send(content)
            
            # Save to database
            await self.save_message(
                message_id=insight_message.id,
                report_id=report.id,
                user_id=None,
                role=MessageRole.BOT.value,
                content=content,
                db=db,
                input_tokens=response_data['input_tokens'],
                output_tokens=response_data['output_tokens'],
                cost_usd=response_data['cost_usd']
            )
            
        except Exception as e:
            print(f"Error sending actionable insight: {e}")

    async def send_qa_invitation(self, message: discord.Message, report: Report, db: Session):
        """Send Q&A invitation as a separate message"""
        try:
            qa_content = "Feel free to ask any questions about your results!"
            qa_message = await message.channel.send(qa_content)
            
            # Save to database
            await self.save_message(
                message_id=qa_message.id,
                report_id=report.id,
                user_id=None,
                role=MessageRole.BOT.value,
                content=qa_content,
                db=db
            )
            
        except Exception as e:
            print(f"Error sending Q&A invitation: {e}")

# Initialize bot instance
biome_bot = BiomeBot()

@bot.event
async def on_ready():
    print(f'✅ {bot.user} is now online and ready!')
    print(f'📊 Servers: {len(bot.guilds)}')

@bot.event
async def on_message(message: discord.Message):
    # Ignore bot messages
    if message.author.bot:
        return
    
    db = SessionLocal()
    try:
        # Handle PDF uploads (when bot is mentioned)
        if bot.user.mentioned_in(message) and message.attachments:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith('.pdf'):
                    if message.author.id in biome_bot.processing_users:
                        await message.reply("⏳ I'm still processing your previous upload. Please wait a moment!")
                        continue
                    
                    biome_bot.processing_users.add(message.author.id)
                    try:
                        await biome_bot.process_pdf_upload(message, attachment, db)
                    finally:
                        biome_bot.processing_users.discard(message.author.id)
                    return
        
        # Handle greeting when mentioned without attachments
        elif bot.user.mentioned_in(message):
            await message.reply("Greetings! 🧬 Upload a microbiome report and we can get started!")
            return
        
        # Handle messages in report threads
        elif isinstance(message.channel, discord.Thread):
            await biome_bot.handle_thread_message(message, db)
    
    except Exception as e:
        print(f"Error handling message: {e}")
        await message.reply("❌ Sorry, I encountered an error. Please try again.")
    
    finally:
        db.close()

@bot.command(name='stats')
async def stats_command(ctx):
    """Show bot statistics"""
    db = SessionLocal()
    try:
        total_reports = db.query(Report).count()
        total_users = db.query(User).count()
        total_messages = db.query(Message).count()
        total_cost = db.query(func.sum(Message.cost_usd)).scalar() or 0
        
        embed = discord.Embed(title="📊 BiomeAI Statistics", color=0x00ff00)
        embed.add_field(name="👥 Users", value=total_users, inline=True)
        embed.add_field(name="📄 Reports", value=total_reports, inline=True)
        embed.add_field(name="💬 Messages", value=total_messages, inline=True)
        embed.add_field(name="💰 Total Cost", value=f"${total_cost:.4f}", inline=True)
        
        await ctx.send(embed=embed)
    
    except Exception as e:
        await ctx.send(f"❌ Error getting stats: {e}")
    finally:
        db.close()

@bot.command(name='health')
async def health_command(ctx):
    """Health check command"""
    await ctx.send("✅ BiomeAI is healthy and ready to analyze microbiome reports!")

def run_bot():
    """Run the Discord bot"""
    if not DISCORD_TOKEN:
        raise ValueError("DISCORD_TOKEN not found in environment variables")
    
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    run_bot()

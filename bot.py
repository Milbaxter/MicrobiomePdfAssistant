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
            # Download PDF
            pdf_bytes = await attachment.read()
            
            # Create thread for this report
            thread = await message.create_thread(
                name=f"🧬 {attachment.filename} - {message.author.display_name}",
                auto_archive_duration=10080  # 7 days
            )
            
            # Process PDF
            await thread.send("📊 Analyzing your microbiome report...")
            
            processed_data = pdf_processor.process_pdf(pdf_bytes)
            
            # Prepare metadata for JSON storage
            metadata_for_db = processed_data['metadata'].copy()
            
            # Extract sample_date_parsed for the database field
            sample_date = metadata_for_db.pop('sample_date_parsed', None)
            
            # Create report record
            report = Report(
                user_id=user.id,
                thread_id=thread.id,
                original_filename=attachment.filename,
                sample_date=sample_date,
                report_metadata=metadata_for_db,
                conversation_stage="awaiting_antibiotics"
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
            
            # Start with date confirmation first
            if sample_date:
                age_months = processed_data['metadata'].get('sample_age_months', 0)
                date_response = f"📅 I see your microbiome report was generated on **{sample_date.strftime('%B %d, %Y')}**\n"
                date_response += f"That's roughly **{age_months} months** ago. Gut profiles can shift fast, so I'll keep that in mind.\n\n"
                date_response += "Did you take any antibiotics around the time of the test?"
            else:
                date_response = "📅 Looks like the report date is missing.\nWhen did you take this test? (Month & year is enough.)"
            
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
        
        # Handle step-by-step conversation flow
        if report.conversation_stage == "awaiting_antibiotics":
            # Store antibiotic response and move to diet prediction
            try:
                updated_metadata = report.report_metadata.copy() if report.report_metadata else {}
                updated_metadata['antibiotics_response'] = message.content
                report.report_metadata = updated_metadata
                report.conversation_stage = "awaiting_diet_prediction"
                db.commit()
                
                # Generate diet prediction
                async with message.channel.typing():
                    await message.channel.send("🔍 Analyzing your gut bacteria patterns...")
                    
                    # Get PDF content chunks for analysis
                    pdf_chunks = await self.find_relevant_chunks("diet microbiome bacteria", report.id, db)
                    pdf_content = "\n".join(pdf_chunks[:3])  # Use top 3 relevant chunks
                    
                    predictions = openai_client.generate_diet_prediction(
                        pdf_content, updated_metadata
                    )
                    
                    # Extract just diet prediction from the response
                    diet_response = "🍽️ **Based on your gut bacteria, I predict you typically eat:**\n\n"
                    diet_response += predictions['content'][:800] + "..." if len(predictions['content']) > 800 else predictions['content']
                    diet_response += "\n\n**Is this accurate?** Tell me about your actual diet and any restrictions you have."
                    
                    bot_message = await message.channel.send(diet_response)
                    
                    await self.save_message(
                        message_id=bot_message.id,
                        report_id=report.id,
                        user_id=None,
                        role=MessageRole.BOT.value,
                        content=diet_response,
                        db=db,
                        input_tokens=predictions['input_tokens'],
                        output_tokens=predictions['output_tokens'],
                        cost_usd=predictions['cost_usd']
                    )
                    return
                    
            except Exception as e:
                print(f"Error in diet prediction stage: {e}")
        
        elif report.conversation_stage == "awaiting_diet_prediction":
            # Store diet response and move to digestive symptoms
            try:
                updated_metadata = report.report_metadata.copy() if report.report_metadata else {}
                updated_metadata['diet_response'] = message.content
                report.report_metadata = updated_metadata
                report.conversation_stage = "awaiting_symptoms_prediction"
                db.commit()
                
                # Generate digestive symptoms prediction
                async with message.channel.typing():
                    # Get PDF content chunks for digestive analysis
                    pdf_chunks = await self.find_relevant_chunks("digestive symptoms bloating bacteria", report.id, db)
                    pdf_content = "\n".join(pdf_chunks[:3])  # Use top 3 relevant chunks
                    
                    digestive_pred = openai_client.generate_digestive_prediction(
                        pdf_content, updated_metadata
                    )
                    
                    symptom_response = "🤢 **Based on your microbiome, I predict you might experience:**\n\n"
                    symptom_response += digestive_pred['content']
                    symptom_response += "\n\n**What digestive symptoms do you actually experience?** (or none if you feel great!)"
                
                bot_message = await message.channel.send(symptom_response)
                
                await self.save_message(
                    message_id=bot_message.id,
                    report_id=report.id,
                    user_id=None,
                    role=MessageRole.BOT.value,
                    content=symptom_response,
                    db=db,
                    input_tokens=digestive_pred['input_tokens'],
                    output_tokens=digestive_pred['output_tokens'],
                    cost_usd=digestive_pred['cost_usd']
                )
                return
                
            except Exception as e:
                print(f"Error in symptoms prediction stage: {e}")
        
        elif report.conversation_stage == "awaiting_symptoms_prediction":
            # Store symptoms response and generate executive summary
            try:
                updated_metadata = report.report_metadata.copy() if report.report_metadata else {}
                updated_metadata['symptoms_response'] = message.content
                report.report_metadata = updated_metadata
                report.conversation_stage = "executive_summary_ready"
                db.commit()
                
                # Generate executive summary with all collected information
                async with message.channel.typing():
                    await message.channel.send("✨ Perfect! Now creating your personalized executive summary...")
                    
                    summary_data = openai_client.generate_executive_summary(
                        "",  # Will use conversation history and RAG
                        updated_metadata
                    )
                    
                    summary_message = await message.channel.send(f"🧬 **EXECUTIVE SUMMARY**\n\n{summary_data['content']}")
                    
                    await self.save_message(
                        message_id=summary_message.id,
                        report_id=report.id,
                        user_id=None,
                        role=MessageRole.BOT.value,
                        content=f"🧬 **EXECUTIVE SUMMARY**\n\n{summary_data['content']}",
                        db=db,
                        input_tokens=summary_data['input_tokens'],
                        output_tokens=summary_data['output_tokens'],
                        cost_usd=summary_data['cost_usd']
                    )
                    
                    followup = await message.channel.send("🎯 **Ready for your questions!** Ask me anything about your microbiome results.")
                    
                    await self.save_message(
                        message_id=followup.id,
                        report_id=report.id,
                        user_id=None,
                        role=MessageRole.BOT.value,
                        content="🎯 **Ready for your questions!** Ask me anything about your microbiome results.",
                        db=db
                    )
                    
                    print(f"💬 Generated executive summary for user {user.username} in report {report.id}")
                    return
                    
            except Exception as e:
                print(f"Error generating executive summary: {e}")
                # Fall through to regular response handling
        
        # Regular conversation handling
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
            
            # Send response
            bot_message = await message.reply(response_data['content'])
            
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
            
            print(f"💬 Responded to user {user.username} in report {report.id}")
            
        except Exception as e:
            print(f"Error generating response: {e}")
            await message.reply("❌ Sorry, I encountered an error processing your question. Please try again.")

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

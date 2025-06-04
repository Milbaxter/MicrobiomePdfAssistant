import json
import os
import numpy as np
from typing import List, Dict, Any
from openai import OpenAI
from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, MAX_CONTEXT_TOKENS,
    EMBEDDING_COST_PER_1K, GPT4O_INPUT_COST_PER_1K, GPT4O_OUTPUT_COST_PER_1K
)

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI's embedding model"""
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def calculate_embedding_cost(self, text: str) -> float:
        """Calculate approximate cost for embedding generation"""
        # Rough token count estimation
        token_count = len(text.split()) * 1.3  # Approximate tokens
        return (token_count / 1000) * EMBEDDING_COST_PER_1K
    
    def calculate_chat_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for chat completion"""
        input_cost = (input_tokens / 1000) * GPT4O_INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * GPT4O_OUTPUT_COST_PER_1K
        return input_cost + output_cost
    
    def count_tokens_rough(self, text: str) -> int:
        """Rough token count estimation"""
        return int(len(text.split()) * 1.3)
    
    def create_microbiome_analysis(self, 
                                 conversation_history: List[Dict[str, str]], 
                                 relevant_chunks: List[str],
                                 user_question: str = None) -> Dict[str, Any]:
        """
        Create microbiome analysis using conversation history and relevant chunks
        
        Args:
            conversation_history: List of previous messages in thread
            relevant_chunks: Relevant PDF chunks from RAG
            user_question: Current user question (if any)
        
        Returns:
            Dict with response content, token usage, and cost
        """
        
        # Build system prompt
        system_prompt = """You are BiomeAI, an expert microbiome analyst assistant. Provide concise, actionable insights from microbiome reports.

Response guidelines:
- Keep responses under 800 characters when possible
- Use bullet points for key findings
- Focus on 2-3 main insights per response
- Be direct and specific
- Ask one focused follow-up question
- Reference specific data from their report

Format your responses:
• Key Finding: [specific insight]
• Recommendation: [actionable step]
• Question: [one relevant follow-up]

Stay concise, accurate, and always suggest consulting healthcare providers for medical decisions."""

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add relevant chunks as context if available
        if relevant_chunks:
            chunks_context = "\n\n".join([f"Report Section: {chunk}" for chunk in relevant_chunks])
            context_message = f"Here are relevant sections from the user's microbiome report:\n\n{chunks_context}"
            messages.append({"role": "system", "content": context_message})
        
        # Add conversation history (map 'bot' role to 'assistant' for OpenAI)
        for msg in conversation_history:
            role = msg["role"]
            if role == "bot":
                role = "assistant"
            messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        # Add current user question if provided
        if user_question:
            messages.append({"role": "user", "content": user_question})
        
        # Estimate total tokens and truncate if necessary
        total_content = system_prompt + str(relevant_chunks) + str(conversation_history) + (user_question or "")
        estimated_tokens = self.count_tokens_rough(total_content)
        
        # Truncate conversation history if too long
        if estimated_tokens > MAX_CONTEXT_TOKENS * 0.8:  # Leave room for response
            print(f"⚠️  Truncating conversation history ({estimated_tokens} tokens)")
            # Keep system messages and recent history
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            recent_messages = [msg for msg in messages if msg["role"] != "system"][-10:]  # Last 10 messages
            messages = system_messages + recent_messages
        
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                max_tokens=400,  # Reduced for more concise responses
                temperature=0.7
            )
            
            # Extract response data
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.calculate_chat_cost(input_tokens, output_tokens)
            
            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost
            }
            
        except Exception as e:
            print(f"Error in OpenAI chat completion: {e}")
            raise

    def generate_executive_summary(self, pdf_content: str, user_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for a new microbiome report"""
        
        # Extract key user info for personalization
        lifestyle_info = []
        if user_metadata.get("diet"):
            lifestyle_info.append(f"Diet: {user_metadata['diet']}")
        if user_metadata.get("age"):
            lifestyle_info.append(f"Age: {user_metadata['age']}")
        if user_metadata.get("height") and user_metadata.get("weight"):
            lifestyle_info.append(f"Height: {user_metadata['height']}, Weight: {user_metadata['weight']}")
        if user_metadata.get("antibiotics"):
            lifestyle_info.append(f"Recent antibiotics: {user_metadata['antibiotics']}")
        
        lifestyle_context = "\n".join(lifestyle_info) if lifestyle_info else "No additional lifestyle information provided."
        
        prompt = f"""Analyze this microbiome report and provide an executive summary with actionable insights.

User Lifestyle Context:
{lifestyle_context}

Microbiome Report Content:
{pdf_content[:4000]}  # Truncate for context limits

Please provide:
1. Key findings from the report
2. Notable patterns or concerns
3. Personalized recommendations based on their lifestyle
4. One specific actionable next step they could consider

Keep the response engaging and supportive, focusing on practical insights."""

        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self.calculate_chat_cost(input_tokens, output_tokens)
            
            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost
            }
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            raise

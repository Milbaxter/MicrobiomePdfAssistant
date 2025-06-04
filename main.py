#!/usr/bin/env python3
"""
BiomeAI Discord Bot - Microbiome PDF Assistant
A Discord bot that analyzes microbiome PDF reports using RAG and OpenAI
"""

import asyncio
import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_database
from bot import run_bot
from config import DISCORD_TOKEN, OPENAI_API_KEY, DATABASE_URL

def check_environment():
    """Check that all required environment variables are set"""
    required_vars = {
        'DISCORD_TOKEN': DISCORD_TOKEN,
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'DATABASE_URL': DATABASE_URL
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables and try again.")
        sys.exit(1)
    
    print("✅ Environment variables configured")

def main():
    """Main entry point"""
    print("🧬 Starting BiomeAI Discord Bot...")
    
    # Check environment
    check_environment()
    
    # Initialize database
    print("📊 Initializing database...")
    try:
        init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        sys.exit(1)
    
    # Start bot
    print("🤖 Starting Discord bot...")
    try:
        run_bot()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

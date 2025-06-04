#!/usr/bin/env python3
"""
Web server wrapper for Discord bot deployment
Provides health check endpoint for Reserved VM deployment
"""

import asyncio
import threading
import sys
import os
from pathlib import Path
from flask import Flask, jsonify

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from database import init_database
from bot import run_bot
from config import DISCORD_TOKEN, OPENAI_API_KEY, DATABASE_URL

app = Flask(__name__)
bot_status = {"running": False, "error": None}

@app.route('/')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        "status": "healthy" if bot_status["running"] else "starting",
        "service": "BiomeAI Discord Bot",
        "message": "Bot is running" if bot_status["running"] else "Bot is starting",
        "error": bot_status["error"]
    })

@app.route('/health')
def health():
    """Additional health endpoint"""
    return jsonify({
        "status": "ok" if bot_status["running"] else "starting",
        "bot_running": bot_status["running"]
    })

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
        return False
    
    print("✅ Environment variables configured")
    return True

def run_discord_bot():
    """Run the Discord bot in a separate thread"""
    try:
        print("🧬 Starting BiomeAI Discord Bot...")
        
        # Check environment
        if not check_environment():
            bot_status["error"] = "Missing environment variables"
            return
            
        # Initialize database
        print("📊 Initializing database...")
        init_database()
        print("✅ Database initialized successfully")
        
        # Start bot
        print("🤖 Starting Discord bot...")
        bot_status["running"] = True
        run_bot()
        
    except Exception as e:
        print(f"Discord bot error: {e}")
        bot_status["error"] = str(e)
        bot_status["running"] = False

if __name__ == "__main__":
    # Start Discord bot in background thread
    bot_thread = threading.Thread(target=run_discord_bot, daemon=True)
    bot_thread.start()
    
    # Run Flask server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
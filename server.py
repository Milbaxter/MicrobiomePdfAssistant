#!/usr/bin/env python3
"""
Web server wrapper for Discord bot deployment
Provides health check endpoint for Reserved VM deployment
"""

import asyncio
import threading
from flask import Flask, jsonify
from main import main as run_bot

app = Flask(__name__)

@app.route('/')
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        "status": "healthy",
        "service": "BiomeAI Discord Bot",
        "message": "Bot is running"
    })

@app.route('/health')
def health():
    """Additional health endpoint"""
    return jsonify({"status": "ok"})

def run_discord_bot():
    """Run the Discord bot in a separate thread"""
    try:
        run_bot()
    except Exception as e:
        print(f"Discord bot error: {e}")

if __name__ == "__main__":
    # Start Discord bot in background thread
    bot_thread = threading.Thread(target=run_discord_bot, daemon=True)
    bot_thread.start()
    
    # Run Flask server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
import os
from dotenv import load_dotenv

load_dotenv()

# Discord Bot Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN_V2") or os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN or DISCORD_TOKEN_V2 environment variable is required")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

# Bot Configuration
BOT_MENTION_NAME = "biomeAI"
MAX_CONTEXT_TOKENS = 16000  # Conservative limit for gpt-4o context window
CHUNK_SIZE = 1000  # Characters per chunk for PDF processing
CHUNK_OVERLAP = 200  # Overlap between chunks
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
MAX_CHUNKS_PER_QUERY = 5  # Maximum relevant chunks to include in context

# Cost tracking (approximate costs per 1K tokens)
EMBEDDING_COST_PER_1K = 0.00002  # text-embedding-3-small
GPT4O_INPUT_COST_PER_1K = 0.0025
GPT4O_OUTPUT_COST_PER_1K = 0.01

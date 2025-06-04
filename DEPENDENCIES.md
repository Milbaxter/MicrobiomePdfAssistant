# BiomeAI Discord Bot Dependencies

## Core Dependencies

The following packages are required for the BiomeAI Discord Bot:

### Discord & API Integration
- `discord-py>=2.5.2` - Discord bot framework
- `openai>=1.84.0` - OpenAI API client for GPT-4o integration

### Database & Vector Search
- `sqlalchemy>=2.0.41` - Database ORM
- `psycopg2-binary>=2.9.10` - PostgreSQL adapter
- `pgvector>=0.4.1` - Vector similarity search extension

### PDF Processing & Data
- `pypdf2>=3.0.1` - PDF text extraction
- `numpy>=2.2.6` - Numerical operations for embeddings

### Configuration
- `python-dotenv>=1.1.0` - Environment variable management

## Installation

Dependencies are managed via the package manager tool in this environment. All packages are automatically installed when running the bot.

## Environment Variables Required

- `DISCORD_TOKEN_V2` - Discord bot token
- `OPENAI_API_KEY` - OpenAI API key for GPT-4o
- `DATABASE_URL` - PostgreSQL connection string (auto-configured)

## Python Version

Requires Python 3.11 or higher.
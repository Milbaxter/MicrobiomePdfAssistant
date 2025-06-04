# BiomeAI Discord Bot

A sophisticated Discord bot that transforms microbiome PDF reports into actionable health insights through an intelligent, step-by-step AI-powered conversation workflow.

## Overview

BiomeAI analyzes microbiome test reports (PDF format) and guides users through a structured conversation to provide personalized health predictions and recommendations. The bot uses advanced RAG (Retrieval-Augmented Generation) with vector embeddings to understand report content and maintain context throughout the conversation.

## Key Features

- **PDF Processing**: Extracts and analyzes microbiome report data from PDF uploads
- **Structured Conversation Flow**: Guides users through a specific sequence of questions and predictions
- **Vector Search**: Uses pgvector for semantic similarity search across report content
- **Cost Tracking**: Monitors OpenAI API usage and costs for each interaction
- **Thread Management**: Creates dedicated Discord threads for each report analysis
- **Automated Follow-ups**: Sends actionable insights and Q&A prompts automatically

## Technology Stack

- **Discord.py**: Bot framework for Discord integration
- **OpenAI GPT-4**: Natural language processing and analysis
- **PostgreSQL + pgvector**: Vector database for embedding storage
- **PyPDF2**: PDF text extraction
- **SQLAlchemy**: Database ORM

## Conversation Flow

The bot follows a specific 8-step conversation sequence:

1. **PDF Upload**: User uploads microbiome report PDF
2. **Date/Antibiotics Questions**: Bot asks about sample date and recent antibiotic use
3. **Diet Prediction**: Short prediction about user's diet based on microbiome data
4. **User Diet Confirmation**: User confirms/corrects diet information and mentions allergies
5. **Energy Prediction**: Short prediction about energy levels
6. **User Energy Confirmation**: User confirms/corrects energy level information
7. **Digestive Prediction**: Short prediction about digestive health
8. **User Digestive Confirmation**: User confirms/corrects digestive symptoms
9. **Executive Summary**: Comprehensive summary combining all data
10. **Automated Follow-ups**: 
    - One actionable insight
    - Q&A invitation

## Prompt Engineering & Dialogue Logic

### 🔍 **Where to Find the Prompts**

The core prompts and conversation logic are located in:

- **`openai_client.py`** - Lines 60-140: Contains all conversation stage prompts
- **`bot.py`** - Lines 340-394: Automated follow-up message logic

### Conversation Stage Detection

The bot determines conversation stages by analyzing:
- Previous bot messages in the thread
- User response patterns
- Specific keywords and phrases

### Key Prompt Locations

1. **Initial Analysis Prompt** (`openai_client.py` lines 65-75):
   - Handles first user questions after PDF upload
   - Asks about sample date and antibiotics

2. **Diet Prediction Prompt** (`openai_client.py` lines 77-90):
   - Creates concise diet predictions (3-4 sentences max)
   - Uses "BE CONCISE" instruction for brevity

3. **Energy Prediction Prompt** (`openai_client.py` lines 92-105):
   - Generates energy level predictions
   - Prompts user for confirmation

4. **Digestive Prediction Prompt** (`openai_client.py` lines 107-115):
   - Predicts digestive health patterns
   - Asks about specific digestive issues

5. **Executive Summary Prompt** (`openai_client.py` lines 117-127):
   - Comprehensive analysis combining all data
   - Must start with "Executive Summary of microbiome report and lifestyle:"

6. **General Q&A Prompt** (`openai_client.py` lines 129-140):
   - Handles follow-up questions
   - Provides detailed explanations when needed

### Automated Follow-up Logic

Located in `bot.py` lines 340-394:

```python
# Detects executive summary and triggers automatic messages
if response_data['content'].lower().startswith('executive summary of microbiome report and lifestyle:'):
    # 1. Generate and send actionable insight
    # 2. Send Q&A invitation
```

## Database Schema

### Tables

- **Users**: Discord user information
- **Reports**: PDF report metadata and conversation tracking
- **ReportChunks**: Vectorized text chunks with embeddings
- **Messages**: Complete conversation history with cost tracking

### Key Features

- Vector similarity search using pgvector
- Conversation stage tracking
- Token usage and cost monitoring
- Thread-based organization

## Setup and Configuration

### Environment Variables Required

```
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql_connection_string
```

### Installation

1. Install dependencies:
```bash
pip install discord.py PyPDF2 python-dotenv sqlalchemy psycopg2-binary pgvector openai
```

2. Set up PostgreSQL with pgvector extension

3. Configure environment variables

4. Run the bot:
```bash
python main.py
```

## Usage

1. Invite the bot to your Discord server
2. Upload a microbiome PDF report in any channel
3. The bot will create a dedicated thread
4. Follow the conversation prompts
5. Receive personalized insights and recommendations

## File Structure

```
├── main.py              # Entry point and environment setup
├── bot.py               # Core Discord bot logic and message handling
├── openai_client.py     # OpenAI integration and prompt management
├── pdf_processor.py     # PDF parsing and text extraction
├── database.py          # Database connection and initialization
├── models.py            # SQLAlchemy database models
└── config.py            # Configuration constants
```

## Monitoring and Debugging

The bot includes comprehensive logging:
- PDF processing status
- Conversation stage transitions
- API costs and token usage
- Error handling and recovery

## Cost Management

- Tracks OpenAI API usage per message
- Estimates embedding generation costs
- Monitors total conversation costs
- Stores cost data in database for analysis

## Contributing

When modifying prompts or conversation logic:

1. **Prompt Changes**: Edit `openai_client.py` lines 60-140
2. **Flow Logic**: Modify conversation detection in `bot.py`
3. **Database Changes**: Update models in `models.py`
4. **Testing**: Use `/health` and `/stats` commands for monitoring

## Prompt Customization Guide

To modify the conversation flow:

1. **Stage Detection**: Update conditions in `openai_client.py`
2. **Prompt Content**: Modify system prompts for each stage
3. **Response Format**: Adjust required response formats
4. **Follow-ups**: Edit automated message content in `bot.py`

The prompts are designed to be:
- Concise (3-4 sentences for predictions)
- Medically informed but accessible
- Contextually aware of previous conversation
- Structured to guide user responses

## License

This project is designed for educational and research purposes in microbiome health analysis.
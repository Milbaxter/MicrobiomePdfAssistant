import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import QueuePool

class Base(DeclarativeBase):
    pass

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def reset_database():
    """Drop all tables and recreate them for a fresh start"""
    from models import User, Report, ReportChunk, Message
    
    print("🔄 Resetting database...")
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped")
    
    # Create pgvector extension
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"⚠️  pgvector extension setup: {e}")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

def init_database():
    """Initialize database tables and extensions"""
    from models import User, Report, ReportChunk, Message
    
    # Create pgvector extension
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"⚠️  pgvector extension setup: {e}")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

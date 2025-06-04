from sqlalchemy import Column, Integer, BigInteger, String, Text, DateTime, Boolean, Numeric, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY
from database import Base
import enum

class MessageRole(enum.Enum):
    USER = "user"
    BOT = "bot"

class User(Base):
    __tablename__ = "users"
    
    id = Column(BigInteger, primary_key=True)  # Discord user ID
    username = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    reports = relationship("Report", back_populates="user")
    messages = relationship("Message", back_populates="user")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=False)
    thread_id = Column(BigInteger, nullable=False, unique=True)  # Discord thread ID
    original_filename = Column(String(255))
    sample_date = Column(DateTime(timezone=True))
    report_metadata = Column(JSON, default=dict)  # Store additional info like user details
    conversation_stage = Column(String(50), default="initial")  # Track conversation flow
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="reports")
    chunks = relationship("ReportChunk", back_populates="report", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="report", cascade="all, delete-orphan")

class ReportChunk(Base):
    __tablename__ = "report_chunks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    chunk_idx = Column(Integer, nullable=False)  # Order of chunk in document
    content = Column(Text, nullable=False)
    # Note: pgvector column will be added dynamically based on available extensions
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    report = relationship("Report", back_populates="chunks")
    
    __table_args__ = (
        Index('report_chunks_report_idx', 'report_id'),
    )

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(BigInteger, primary_key=True)  # Discord message ID
    report_id = Column(Integer, ForeignKey("reports.id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("users.id"), nullable=True)  # Null for bot messages
    role = Column(String(10), nullable=False)  # 'user' or 'bot'
    content = Column(Text, nullable=False)
    
    # Token and cost tracking
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Numeric(10, 6), default=0)
    
    # RAG tracking
    retrieved_chunk_ids = Column(ARRAY(Integer), default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    report = relationship("Report", back_populates="messages")
    user = relationship("User", back_populates="messages")
    
    __table_args__ = (
        Index('messages_report_created_idx', 'report_id', 'created_at'),
        Index('messages_thread_order_idx', 'report_id', 'id'),
    )

# Add pgvector column if extension is available
try:
    from pgvector.sqlalchemy import Vector
    
    # Add embedding column to ReportChunk
    ReportChunk.embedding = Column(Vector(1536))  # OpenAI embedding dimension
    
    # Create vector index
    ReportChunk.__table_args__ = (
        Index('report_chunks_report_idx', 'report_id'),
        Index('report_chunks_embedding_idx', 'embedding', postgresql_using='ivfflat',
              postgresql_with={'lists': 100}, postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
    
    print("✅ pgvector columns and indexes configured")
    
except ImportError:
    print("⚠️  pgvector not available, using fallback storage")
    # Fallback to array storage
    ReportChunk.embedding_array = Column(ARRAY(Numeric), nullable=True)

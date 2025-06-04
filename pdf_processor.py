import PyPDF2
import io
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import CHUNK_SIZE, CHUNK_OVERLAP

class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF bytes"""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    text_content.append(text)
            
            full_text = "\n\n".join(text_content)
            return self.clean_text(full_text)
            
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            raise Exception(f"Failed to process PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common PDF artifacts
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'Page \d+', '', text)
        
        # Fix common OCR issues
        text = text.replace('_', ' ')
        
        return text.strip()
    
    def extract_sample_date(self, text: str) -> Optional[datetime]:
        """Extract sample date from PDF text"""
        # Common date patterns in microbiome reports
        date_patterns = [
            r'sample.*?date.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'collected.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'test.*?date.*?(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',  # Generic date
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                date_str = matches[0]
                try:
                    # Try different date formats
                    for fmt in ['%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%d/%m/%Y', '%m/%d/%y']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
                except Exception:
                    continue
        
        return None
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for RAG"""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            # Calculate end position
            end = start + CHUNK_SIZE
            
            # If we're not at the end, try to break at a sentence or paragraph
            if end < len(text):
                # Look for sentence break
                sentence_break = text.rfind('.', start, end)
                if sentence_break > start + CHUNK_SIZE // 2:
                    end = sentence_break + 1
                else:
                    # Look for paragraph break
                    para_break = text.rfind('\n', start, end)
                    if para_break > start + CHUNK_SIZE // 2:
                        end = para_break
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunks.append({
                    'chunk_idx': chunk_idx,
                    'content': chunk_content,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_idx += 1
            
            # Move start position with overlap
            start = max(start + CHUNK_SIZE - CHUNK_OVERLAP, end)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from PDF content"""
        metadata = {}
        
        # Extract sample date
        sample_date = self.extract_sample_date(text)
        if sample_date:
            # Convert datetime to ISO string for JSON serialization
            metadata['sample_date'] = sample_date.isoformat()
            # Calculate age of sample
            now = datetime.now()
            age_months = (now.year - sample_date.year) * 12 + (now.month - sample_date.month)
            metadata['sample_age_months'] = age_months
        
        # Extract other potential metadata
        # Look for lab name
        lab_patterns = [
            r'(Viome|Thryve|uBiome|Gut Intelligence|Microba)',
        ]
        for pattern in lab_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metadata['lab_name'] = matches[0]
                break
        
        # Look for diversity metrics
        diversity_patterns = [
            r'shannon.*?diversity.*?(\d+\.?\d*)',
            r'simpson.*?index.*?(\d+\.?\d*)',
        ]
        for pattern in diversity_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                metadata['diversity_metrics'] = matches
                break
        
        return metadata
    
    def process_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Complete PDF processing pipeline"""
        try:
            # Extract text
            text_content = self.extract_text_from_pdf(pdf_bytes)
            
            if not text_content:
                raise Exception("No text content found in PDF")
            
            # Extract metadata
            metadata = self.extract_metadata(text_content)
            
            # Create chunks
            chunks = self.chunk_text(text_content)
            
            return {
                'text_content': text_content,
                'metadata': metadata,
                'chunks': chunks,
                'total_chars': len(text_content),
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

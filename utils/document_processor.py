from typing import List, Dict
import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': TextLoader
        }

    def process_file(self, file_path: str) -> List[Dict[str, str]]:
        """Process a file and return chunks with metadata"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.loaders:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            loader = self.loaders[file_ext](file_path)
            documents = loader.load()
            
            # Convert to dict format with metadata
            chunks = []
            for doc in documents:
                chunks.append({
                    'text': doc.page_content,
                    'metadata': {
                        'source': os.path.basename(file_path),
                        **doc.metadata
                    }
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
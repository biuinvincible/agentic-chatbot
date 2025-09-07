"""
File Processing Workflow for the Agentic Chatbot.

This module handles the complete workflow for document processing:
1. File upload
2. Text extraction
3. Embedding generation
4. Vector store creation
5. Query processing
"""
import os
import uuid
import time
from typing import Dict, Any, Optional, Callable
from agents.file_processing.core import process_file, get_vector_store_by_session

class DocumentProcessor:
    """Handles the complete document processing workflow."""
    
    def __init__(self, session_id: str, progress_callback: Callable = None):
        """
        Initialize the document processor with a session ID.
        
        Args:
            session_id (str): Session ID for this processing session
            progress_callback (Callable, optional): Callback function for progress updates
        """
        self.session_id = session_id
        self.processed_documents = {}
        self.progress_callback = progress_callback
    
    def _update_progress(self, message: str, percent: int = None):
        """
        Update progress if callback is provided.
        
        Args:
            message (str): Progress message
            percent (int, optional): Completion percentage
        """
        if self.progress_callback:
            self.progress_callback(message, percent)
    
    def upload_and_process(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and process a document file with progress updates.
        
        Args:
            file_path (str): Path to the uploaded file.
            
        Returns:
            Dict[str, Any]: Processing result with status and document info.
        """
        self._update_progress("Starting document processing...", 0)
        
        if not os.path.exists(file_path):
            self._update_progress("Error: File not found", 100)
            return {"status": "error", "message": "File not found"}
        
        # Get file name for progress messages
        file_name = os.path.basename(file_path)
        self._update_progress(f"Processing {file_name}...", 10)
        
        # Process the file (extract text, chunk, embed, create vector store)
        self._update_progress(f"Extracting text from {file_name}...", 20)
        file_info = process_file(file_path, self.session_id, max_chunk_size=10000)
        
        if "error" in file_info:
            self._update_progress(f"Error processing {file_name}: {file_info['error']}", 100)
            return {"status": "error", "message": file_info["error"]}
        
        self._update_progress(f"Processing {file_name} completed", 90)
        
        # Store document info for later retrieval
        doc_id = str(uuid.uuid4())
        has_vector_store = file_info.get("vector_store") is not None
        document_info = {
            "doc_id": doc_id,
            "file_name": file_info["file_name"],
            "file_path": file_info["file_path"],
            "chunk_count": file_info.get("chunk_count", 0),
            "has_vector_store": has_vector_store,
            "content_preview": file_info["content"][:500] + "..." if len(file_info["content"]) > 500 else file_info["content"],
            "processing_complete": True,  # Processing is complete whether or not we have a vector store
            "is_image": file_info.get("is_image", False)  # Include image flag for proper routing
        }
        
        self.processed_documents[doc_id] = document_info
        
        self._update_progress(f"Document '{file_info['file_name']}' ready", 100)
        
        # Return success status with document info
        return {
            "status": "success",
            "message": f"Document '{file_info['file_name']}' processed successfully",
            "document_info": document_info,
            "processing_log": file_info.get("processing_log", [])
        }
    
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve information about a processed document.
        
        Args:
            doc_id (str): Document ID.
            
        Returns:
            Optional[Dict[str, Any]]: Document information or None if not found.
        """
        return self.processed_documents.get(doc_id)
    
    def list_documents(self) -> Dict[str, Any]:
        """
        List all processed documents in this session.
        
        Returns:
            Dict[str, Any]: Dictionary of document info keyed by doc_id.
        """
        return self.processed_documents
    
    def is_document_processed(self, doc_id: str) -> bool:
        """
        Check if a document has been fully processed and embedded.
        
        Args:
            doc_id (str): Document ID.
            
        Returns:
            bool: True if document is processed, False otherwise.
        """
        doc_info = self.processed_documents.get(doc_id)
        return doc_info is not None and doc_info.get("has_vector_store", False)
    
    def get_vector_store(self, doc_id: str):
        """
        Get the vector store for a processed document.
        
        Args:
            doc_id (str): Document ID.
            
        Returns:
            Chroma vector store or None if not available.
        """
        doc_info = self.processed_documents.get(doc_id)
        if not doc_info:
            return None
            
        try:
            return get_vector_store_by_session(self.session_id)
        except Exception as e:
            print(f"Error getting vector store: {str(e)}")
            return None
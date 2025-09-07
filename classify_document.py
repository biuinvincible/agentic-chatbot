"""
Document classification module for the Agentic Assistant.
Analyzes document characteristics to determine optimal processing strategy.
"""

import os
from typing import Dict, Any

def classify_document_for_processing(file_path: str) -> Dict[str, Any]:
    """
    Analyze a document to determine its characteristics and recommend processing strategy.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary containing document analysis and processing recommendations
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        # Get basic file info
        file_size = os.path.getsize(file_path)
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Simple extension-based classification
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.csv', '.xml', '.json'}
        document_extensions = {'.pdf', '.docx'}
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        
        # Default classification values
        doc_type = "text_only"
        use_vlm = False
        strategy = "fast_text"
        reason = "Default classification for file type"
        
        # Classify based on extension
        if ext in text_extensions:
            doc_type = "text_only"
            use_vlm = False
            strategy = "fast_text"
            reason = f"Plain text file extension {ext} requires no visual analysis"
        elif ext in document_extensions:
            doc_type = "document"
            use_vlm = True  # Assume documents might contain images/tables
            strategy = "selective_vlm"
            reason = f"Document file extension {ext} may contain visual elements"
        elif ext in image_extensions:
            doc_type = "image"
            use_vlm = True
            strategy = "full_vlm"
            reason = f"Image file extension {ext} requires visual analysis"
        else:
            doc_type = "unknown"
            use_vlm = False
            strategy = "fast_text"
            reason = "Unknown file type, using conservative text-only processing"
            
        # For PDF files, do some basic content analysis
        if ext == '.pdf':
            try:
                # This would be a more sophisticated analysis in a full implementation
                # For now, we'll use heuristics based on file size
                if file_size > 10 * 1024 * 1024:  # 10MB
                    # Large PDFs likely have complex content
                    use_vlm = True
                    strategy = "selective_vlm"
                    reason = "Large PDF file likely contains visual elements"
                elif file_size < 1024:  # 1KB
                    # Very small PDFs are likely text-only
                    use_vlm = False
                    strategy = "fast_text"
                    reason = "Small PDF file likely contains text-only content"
            except Exception:
                # If analysis fails, fall back to defaults
                pass
                
        return {
            "file_extension": ext,
            "file_size": file_size,
            "doc_type": doc_type,
            "use_vlm": use_vlm,
            "strategy": strategy,
            "reason": reason
        }
        
    except Exception as e:
        return {
            "error": f"Error classifying document: {str(e)}",
            "doc_type": "unknown",
            "use_vlm": False,
            "strategy": "fast_text",
            "reason": "Error in classification, using conservative approach"
        }

def quick_document_analysis(file_path: str) -> Dict[str, Any]:
    """
    Quick analysis to determine if document needs special processing.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with basic document characteristics
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
            
        # Get basic file info
        file_size = os.path.getsize(file_path)
        _, ext = os.path.splitext(file_path)
        
        # Simple extension-based classification
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css'}
        document_extensions = {'.pdf', '.docx'}
        
        if ext.lower() in text_extensions:
            doc_type = "plain_text"
            use_vlm = False
        elif ext.lower() in document_extensions:
            doc_type = "document"
            use_vlm = True  # Might contain images/tables
        else:
            doc_type = "unknown"
            use_vlm = False
            
        return {
            "file_size": file_size,
            "extension": ext.lower(),
            "doc_type": doc_type,
            "use_vlm": use_vlm,
            "needs_visual_analysis": use_vlm
        }
        
    except Exception as e:
        return {"error": f"Error analyzing document: {str(e)}"}
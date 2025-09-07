"""
Docling-based document processing for the Agentic Chatbot.
This module implements a hybrid approach for document processing:
1. Fast text extraction using PyPDF2/docx for main content
2. Selective VLM processing using Ollama's granite3.2-vision for figures/tables when needed
"""

import os
from typing import Dict, Any

def extract_text_from_pdf_docling(file_path: str) -> str:
    """
    Extract text from a PDF file using Docling with VLM processing.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content with VLM processing for visual elements
    """
    return process_document_with_vlm(file_path)

def extract_text_from_docx_docling(file_path: str) -> str:
    """
    Extract text from a DOCX file using Docling with VLM processing.
    
    Args:
        file_path (str): Path to the DOCX file
        
    Returns:
        str: Extracted text content with VLM processing for visual elements
    """
    return process_document_with_vlm(file_path)

def process_document_with_vlm(file_path: str) -> str:
    """
    Process document with VLM processing for visual elements using accelerator options.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Processed document content
    """
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        
        # Configure pipeline with Ollama VLM for picture description
        pipeline_options = PdfPipelineOptions()
        
        # Add accelerator options with 24 threads on CPU
        accelerator_options = AcceleratorOptions(
            num_threads=24, device=AcceleratorDevice.CPU
        )
        pipeline_options.accelerator_options = accelerator_options
        
        # Enable remote services and picture description with Ollama
        pipeline_options.enable_remote_services = True
        pipeline_options.do_picture_description = True
        
        # Configure Ollama VLM options using PictureDescriptionApiOptions
        vlm_options = PictureDescriptionApiOptions(
            url="http://localhost:11434/v1/chat/completions",
            params=dict(
                model="qwen2.5vl:3b",
            ),
            prompt="""You are an image annotator. 
            Your task is to produce a plain, exhaustive, and faithful description of the given image. 
            Describe exactly what is present in the image, without any hallucination, interpretation, or inference. 
            Do not add information that is not explicitly visible. 
            Be as detailed as possible, and do not omit any visual element, no matter how small. 
            Just describe what you see, what you think it is is not included.
            If the image needs to be interpreted in direction (left to right, top to bottom,... etc) then interpret it in direction for better clarity.

            If the image is a table, describe the table structure, rows, columns, headers, and all cell values.  
            If the image is a figure, chart, or diagram, describe precisely what it shows, including labels, axes, legends, and any textual or graphical elements.  
            If it contains text, transcribe it exactly as written.  

            Output only the detailed description of the image, nothing else.""",
            timeout=3000,
        )
        pipeline_options.picture_description_options = vlm_options
        
        # Keep other processing reasonable
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        
        # Determine file type
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        input_format = InputFormat.PDF if ext == ".pdf" else InputFormat.DOCX
        
        converter = DocumentConverter(
            format_options={
                input_format: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        
        # Convert the document
        result = converter.convert(file_path)
        
        # Export to markdown format
        markdown_content = result.document.export_to_markdown()
        
        return markdown_content
    except Exception as e:
        # Fallback to fast processing if VLM fails
        print(f"VLM processing failed, falling back to fast processing: {e}")
        return process_document_without_vlm(file_path)

def process_document_without_vlm(file_path: str) -> str:
    """
    Process document without VLM for maximum speed.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Fast processed document content
    """
    import PyPDF2
    import docx
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".pdf":
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    elif ext == ".docx":
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += f"{paragraph.text}\n"
            return text
        except Exception as e:
            return f"Error extracting text from DOCX: {str(e)}"
    else:
        # For text files
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
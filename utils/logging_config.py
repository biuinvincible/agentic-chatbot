"""
Logging configuration for the Agentic Assistant project.
"""
import logging
import os

def setup_logging():
    """Set up logging configuration for the project."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific log levels for different modules
    logging.getLogger('langchain').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def get_logger(name: str):
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
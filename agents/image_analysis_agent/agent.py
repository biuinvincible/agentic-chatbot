from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import base64
import os
# Import unified state
from agents.state import UnifiedState

# Define the image analysis prompt
image_analysis_prompt = PromptTemplate.from_template(
    "You are the 'Image Analysis Agent', a specialized agent for analyzing images provided by the user.\n"
    "Your core capabilities are:\n"
    "1. Analyzing images provided by the user.\n"
    "2. Answering questions about the content of images.\n"
    "3. Describing the visual elements present in an image.\n"
    "4. Identifying objects, characters, symbols, or items in images.\n\n"
    "Your limitations and protocols:\n"
    "- Focus on providing accurate and detailed descriptions of the image content.\n"
    "- If you cannot analyze the image for any reason, explain why clearly.\n"
    "- Always provide a comprehensive response based on the image analysis.\n"
    "- If the user is asking about a specific item (like a gun skin, character, etc.), focus on identifying and describing that item.\n\n"
    "Supervisor's Task Guidance: {task_guidance}\n\n"
    "Your goal is to provide a thorough analysis of the image to help the user understand its content, following the supervisor's specific guidance above.\n\n"
    "The user wants you to: '{task}'\n\n"
    "Analyze the provided image and respond to the user's request with specific attention to any items they're asking about and the supervisor's guidance."
)

def create_image_message_content(task: str, image_file_path: str):
    """
    Create message content for image analysis with actual image data.
    
    Args:
        task (str): The user's request about the image
        image_file_path (str): Path to the image file
        
    Returns:
        list: Message content with text and image data
    """
    # Check if file exists
    if not os.path.exists(image_file_path):
        return f"Error: Image file not found at {image_file_path}"
    
    # Get file extension
    _, ext = os.path.splitext(image_file_path)
    ext = ext.lower().lstrip('.')
    
    # Map extension to MIME type
    mime_types = {
        'jpg': 'jpeg',
        'jpeg': 'jpeg',
        'png': 'png',
        'gif': 'gif',
        'bmp': 'bmp',
        'tiff': 'tiff',
        'webp': 'webp'
    }
    
    mime_type = mime_types.get(ext, 'jpeg')  # Default to jpeg
    
    try:
        # Read and encode the image file
        with open(image_file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Create the message content with both text and image
        return [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{encoded_image}"}}
        ]
    except Exception as e:
        return f"Error reading image file: {str(e)}"

# Define the image analysis agent
def image_analysis_agent(state: UnifiedState, llm) -> UnifiedState:
    """
    Image Analysis Agent that analyzes images provided by the user.
    """
    print("[ImageAnalysis] Activated")
    
    # Get the supervisor's task guidance
    current_task = state.get("current_task", {})
    task_description = current_task.get("description", "")
    task_guidance = f"Task: {task_description}" if task_description else "No specific task guidance provided"
    
    # Get the last message (user's task)
    last_message = state["messages"][-1]
    task = last_message.content if hasattr(last_message, 'content') else str(last_message)
    print("[ImageAnalysis] Processing image task...")
    
    # Get the last 6 messages for context (excluding the current message)
    conversation_context_messages = state["messages"][-7:-1]  # Last 6 messages excluding current
    
    # Check if we have document info with an image file
    document_info = state.get("document_info", [])
    image_file_path = None
    
    # Look for image files in document info
    # Check for explicit is_image flag or implicit indicators (chunk_count=0, has_vector_store=False)
    print("[ImageAnalysis] Checking for image documents...")
    for doc in document_info:
        is_image_file = (
            doc.get("is_image", False) or 
            (doc.get("chunk_count", -1) == 0 and doc.get("has_vector_store", True) == False)
        )
        if is_image_file and "file_path" in doc:
            image_file_path = doc["file_path"]
            print(f"[ImageAnalysis] Found image file: {image_file_path}")
            break
    
    # If we found an image file, create proper multimodal content
    if image_file_path:
        # Check if file exists
        if not os.path.exists(image_file_path):
            print(f"Image file not found at path: {image_file_path}")
            response_text = f"Image Analysis Agent Error:\nI detected that you've uploaded an image file, but I couldn't find it at the expected location ({image_file_path}). This might be because the file was cleaned up after upload. Please try uploading the image again and ask your question immediately after uploading."
            print(f"Returning error response: {response_text}")
            return {"messages": state["messages"] + [HumanMessage(content=response_text, name="ImageAnalysisAgent")]}
            
        print(f"Processing image file: {image_file_path}")
        # Create a more general task if the user is just asking about the image file
        if "what is" in task.lower() and any(ext in task.lower() for ext in [".jpg", ".png", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]):
            analysis_task = "Please analyze this image and describe what you see in detail."
        else:
            analysis_task = task
            
        print("[ImageAnalysis] Creating image message content")
        image_content = create_image_message_content(analysis_task, image_file_path)
        if isinstance(image_content, list):
            print("[ImageAnalysis] Sending image to LLM for analysis...")
            # Create a new message with multimodal content
            image_message = HumanMessage(content=image_content)
            # Generate analysis using the LLM
            analysis_result = llm.invoke([image_message])
            print("[ImageAnalysis] Received analysis result from LLM")
        else:
            # Error occurred, return error message
            print(f"[ImageAnalysis] Error creating image content: {image_content}")
            analysis_result = type('obj', (object,), {'content': image_content})()
    else:
        print("[ImageAnalysis] No valid image file found")
        # No image file found - provide a more helpful response
        if document_info:
            response_text = f"Image Analysis Agent Error:\nI detected that you've uploaded documents, but I couldn't find any image files to analyze. The uploaded documents appear to be: {[doc.get('file_name', 'Unknown') for doc in document_info]}. If you intended to analyze an image, please make sure you've uploaded an image file (PNG, JPG, JPEG, etc.) and try again."
        else:
            response_text = "Image Analysis Agent Error:\nNo documents have been uploaded for analysis. To analyze an image, please upload an image file (PNG, JPG, JPEG, etc.) first, then ask your question about it."
        print("[ImageAnalysis] Returning error response")
        return {"messages": state["messages"] + [HumanMessage(content=response_text, name="ImageAnalysisAgent")]}
    
    # Format the response
    response_text = f"Image Analysis Result:\n{analysis_result.content if hasattr(analysis_result, 'content') else str(analysis_result)}"
    print(f"Returning analysis result: {response_text}")
    
    # --- NEW: Report Outcome ---
    outcome_info = {
        "agent": "image_analysis_agent",
        "status": "completed",  # Assume success if we reach here. Could refine based on errors.
        "details": "Image analysis completed successfully"
    }
    # --- END NEW ---
    
    # Add the result to the state and return control to supervisor
    return {
        "messages": state["messages"] + [HumanMessage(content=response_text, name="ImageAnalysisAgent")],
        "last_agent_outcome": outcome_info,  # Add the outcome information
        "next": "supervisor"  # Return control to supervisor
    }
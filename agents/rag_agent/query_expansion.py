"""
Query Expansion module for RAG Agent to improve document retrieval.
"""
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the query expansion prompt specifically for RAG
query_expansion_prompt = PromptTemplate.from_template(
    "You are an AI assistant tasked with expanding a user's question to improve document retrieval in a RAG system.\n"
    "The original question is: '{question}'\n\n"
    "Please provide 3 expanded versions of this question that would help retrieve more relevant document chunks.\n"
    "Consider:\n"
    "- Synonyms and related terms\n"
    "- Different ways the same information might be expressed\n"
    "- Broader or more specific versions of the question\n"
    "- Context that might help identify relevant passages\n\n"
    "Format your response as a numbered list:\n"
    "1. [expanded question 1]\n"
    "2. [expanded question 2]\n"
    "3. [expanded question 3]\n"
    "Expanded questions:\n"
)

def expand_query_for_rag(question: str, llm) -> List[str]:
    """
    Expand a query using an LLM to improve RAG retrieval performance.
    
    Args:
        question (str): The original user question
        llm: The language model to use for expansion
        
    Returns:
        List[str]: List of expanded queries including the original
    """
    try:
        # Create the expansion chain
        expansion_chain = query_expansion_prompt | llm | StrOutputParser()
        expansion_result = expansion_chain.invoke({"question": question})
        
        # Parse expanded queries (simple parsing for numbered list)
        expanded_queries = []
        for line in expansion_result.split('\n'):
            if line.strip().startswith(('1.', '2.', '3.')):
                query = line.split('.', 1)[1].strip()
                if query:
                    expanded_queries.append(query)
                    
        # Fallback: if parsing fails, use basic expansion
        if not expanded_queries:
            expanded_queries = [
                question,
                f"{question} details",
                f"{question} explanation"
            ]
        # Add the original query if not already present
        elif question not in expanded_queries:
            expanded_queries.insert(0, question)
            
        return expanded_queries
    except Exception as e:
        print(f"Error during query expansion: {e}")
        # Fallback to original query only
        return [question]
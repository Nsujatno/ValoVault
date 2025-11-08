import os

from openai import OpenAI
from typing import List, Union, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    if model is None:
        model = EMBEDDING_MODEL
    try:
        # clean text
        text = text.replace("\n", " ").strip()

        # call model
        response = client.embeddings.create(
            model=model,
            input=text
        )

        # extract embedding vector
        embedding = response.data[0].embedding

        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def create_play_embedding(play_data: dict) -> List[float]:
    text_parts = [
        f"Map: {play_data['map']}",
        f"Agent: {play_data['agent']}"
    ]

    if play_data.get('enemy_agent'):
        text_parts.append(f"Enemy Agent: {play_data['enemy_agent']}")
    
    text_parts.append(f"Play: {play_data['play_description']}")
    
    # Combine all parts
    combined_text = " | ".join(text_parts)
    
    return get_embedding(combined_text)

def create_query_embedding(query: str, context: Optional[dict] = None) -> List[float]:
    if context:
        # Add context to query for better matching
        text_parts = []
        
        if context.get('map'):
            text_parts.append(f"Map: {context['map']}")
        if context.get('agent'):
            text_parts.append(f"Agent: {context['agent']}")
        if context.get('enemy_agent'):
            text_parts.append(f"Enemy Agent: {context['enemy_agent']}")
        
        text_parts.append(f"Query: {query}")
        combined_text = " | ".join(text_parts)
        
        return get_embedding(combined_text)
    
    return get_embedding(query)
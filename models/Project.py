from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    """
     include session_id and user_id so we can uniquely track 
    each user's conversation history in  a database.
    
    conversation_history remains for backward compatibility if needed.
    """
    session_id: str
    user_id: str
    user_query: str
    conversation_history: Optional[str] = ""
    car_details: Optional[str] = ""

class ChatResponse(BaseModel):
    assistant_response: str

class ImageUploadResponse(BaseModel):
    car_details: str

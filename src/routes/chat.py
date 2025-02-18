from fastapi import APIRouter
from models import ChatRequest, ChatResponse
from controllers.bb import ChatbotController

chat_router = APIRouter()
chatbot = ChatbotController()

@chat_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    This endpoint handles user chat requests. The POST body should include:
      - session_id (str): A unique identifier for the session.
      - user_id (str): A unique identifier for the user.
      - user_query (str): The user's query or message.
      - conversation_history (str, optional): Existing conversation context.
      - car_details (str, optional): Additional details extracted from an image.
    
    The method retrieves any existing conversation history from the ChatbotController,
    then calls the ReAct agent to generate an appropriate response. Finally, 
    it appends the latest user message and the generated response to the conversation history.
    """
    # Retrieve any existing conversation history from in-memory store
    existing_history = chatbot.get_conversation_history(
        session_id=request.session_id,
        user_id=request.user_id
    )

    # Pass the user query to the ReAct agent for response generation
    response_text = chatbot.react_agent(
        user_prompt=request.user_query,
        conversation_history=(existing_history or request.conversation_history),
        car_details=request.car_details
    )

    # Store the updated conversation in the controller's in-memory store
    chatbot.append_to_history(
        session_id=request.session_id,
        user_id=request.user_id,
        user_text=request.user_query,
        assistant_text=response_text
    )

    return ChatResponse(assistant_response=response_text)

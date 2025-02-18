from typing import Dict, Tuple, Optional, List
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException

from .SQL_AgentController import SQL_AgentController
from .BaseController import BaseController
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.PromptTemplate import get_prompt_template


class ChatbotController(BaseController):
    """
    The ChatbotController is responsible for managing user conversations,
    interacting with the text generation model (LLM), processing images,
    and handling SQL queries through an SQL agent.
    """

    def __init__(self) -> None:
        """
        Initialize all required components for the Chatbot, including 
        LLM, Vision, and SQL models, as well as an in-memory conversation store.
        """
        super().__init__()

        # In-memory conversation store, keyed by (session_id, user_id).
        # The value is a string representing the full conversation so far.
        self.conversation_store: Dict[Tuple[str, str], str] = {}

        # Load prompt templates.
        self.prompt_template = get_prompt_template()

        # Set up the LLM provider factory.
        self.llm_provider_factory = LLMProviderFactory(self.app_settings)

        # Text generation client.
        self.text_generation_client = self.llm_provider_factory.create(
            provider=self.app_settings.GENERATION_BACKEND
        )
        self.text_generation_client.set_generation_model(
            model_id=self.app_settings.GENERATION_MODEL_ID
        )

        # Vision client (for image processing).
        self.vision_client = self.llm_provider_factory.create(
            provider=self.app_settings.VISION_BACKEND
        )
        self.vision_client.set_vision_model(
            model_id=self.app_settings.VISION_MODEL_ID
        )

        # Text generation client for SQL queries.
        self.text_generation_client_sql = self.llm_provider_factory.create(
            provider=self.app_settings.SQL_BACKEND
        )
        self.text_generation_client_sql.set_generation_model(
            model_id=self.app_settings.SQL_MODEL_ID
        )
        self.llm_sql = self.text_generation_client_sql.LLM_CHAT()

        # Initialize the SQL_AgentController.
        self.sql_agent = SQL_AgentController(self.llm_sql)

        # ReAct system prompt from the prompt template.
        self.react_system_prompt: str = self.prompt_template.react_system_prompt()

    def get_conversation_history(self, session_id: str, user_id: str) -> str:
        """
        Retrieve the conversation history from the in-memory store, if it exists.

        :param session_id: The session identifier.
        :param user_id: The user identifier.
        :return: The conversation text, or an empty string if none is found.
        """
        key = (session_id, user_id)
        return self.conversation_store.get(key, "")

    def append_to_history(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        assistant_text: str
    ) -> None:
        """
        Append the user's last message and the assistant's response to the conversation history.

        :param session_id: The session identifier.
        :param user_id: The user identifier.
        :param user_text: The text of the user's message.
        :param assistant_text: The text of the assistant's response.
        """
        key = (session_id, user_id)
        existing_history = self.conversation_store.get(key, "")

        updated_history = (
            f"{existing_history}\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text}"
        )
        self.conversation_store[key] = updated_history

    def process_uploaded_image(self, file: UploadFile) -> str:
        """
        Process the uploaded image to analyze and extract relevant details
        using the vision client.

        :param file: The uploaded image file.
        :return: A string containing the extracted details.
        """
        try:
            car_details = self.vision_client.vision_to_text(file)
            return car_details
        except Exception as e:
            logging.error(f"Error analyzing the image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing the image: {str(e)}"
            ) from e

    def handle_sql_mode(self, user_prompt: str) -> str:
        """
        Handle SQL-related queries through the SQL_AgentController.

        :param user_prompt: The user's prompt or query.
        :return: The assistant's response after executing the SQL query.
        """
        try:
            assistant_response = self.sql_agent.chat_agent_with_sql(user_prompt)
            return assistant_response
        except Exception as e:
            logging.error(f"Error in SQL mode: {str(e)}")
            return f"Error generating SQL response: {str(e)}"

    def react_agent(
        self,
        user_prompt: str,
        conversation_history: str = "",
        car_details: str = ""
    ) -> str:
        """
        Execute the ReAct Agent approach to handle the user's query.
        It iterates through the pattern: Thought -> Action -> Observation -> Answer.

        :param user_prompt: The user's input text.
        :param conversation_history: The previous conversation text, if any.
        :param car_details: Details extracted from an image, if any.
        :return: The final answer, or a fallback message if no answer is found.
        """
        # Prepare the message list for the LLM.
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": self.react_system_prompt})

        if conversation_history:
            messages.append({
                "role": "assistant",
                "content": f"Conversation history: {conversation_history}"
            })

        if car_details:
            messages.append({
                "role": "assistant",
                "content": f"Car image details: {car_details}"
            })

        # Include the user's prompt.
        messages.append({"role": "user", "content": user_prompt})

        max_iterations: int = 3
        for _ in range(max_iterations):
            assistant_reply = self.text_generation_client.generate_text(
                prompt=self.prompt_template(user_prompt),
                chat_history=messages
            )

            # Add the assistant's reply to the message list.
            messages.append({"role": "assistant", "content": assistant_reply})

            # Check if the reply contains the final answer.
            if "Answer:" in assistant_reply:
                final_answer = assistant_reply.split("Answer:", 1)[1].strip()
                return final_answer

            # Check if the reply contains an Action step.
            if "Action:" in assistant_reply:
                # Expected format: "Action: handle_sql_mode: <INPUT>"
                action_part = assistant_reply.split("Action:", 1)[1].strip()

                if ":" in action_part:
                    tool_name, tool_input = action_part.split(":", 1)
                    tool_name = tool_name.strip()
                    tool_input = tool_input.strip()

                    observation_result: Optional[str] = None
                    if tool_name == "handle_sql_mode":
                        observation_result = self.handle_sql_mode(tool_input)
                    elif tool_name == "process_uploaded_image":
                        # For a real application, the actual file would need to be passed.
                        observation_result = f"(Mocked) Called process_uploaded_image with: {tool_input}"
                    else:
                        observation_result = f"Unknown tool: {tool_name}"

                    messages.append({
                        "role": "system",
                        "content": f"Observation: {observation_result}"
                    })
                else:
                    messages.append({
                        "role": "system",
                        "content": "Observation: Could not parse Action properly."
                    })
            else:
                # If there is no Action, continue until we find an answer or reach the iteration limit.
                continue

        return "I'm sorry, but I couldn't find a final answer."

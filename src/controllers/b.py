from .SQL_AgentController import SQL_AgentController
from .BaseController import BaseController
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.llm.PromptTemplate import get_prompt_template
from fastapi import FastAPI, UploadFile, File, HTTPException
import logging

class ChatbotController(BaseController):
    def __init__(self):
        super().__init__()

        # Load Prompt Template
        self.prompt_template = get_prompt_template()

        # Set up LLM/Vision Clients
        self.llm_provider_factory = LLMProviderFactory(self.app_settings)

        # Text generation client
        self.text_generation_client = self.llm_provider_factory.create(
            provider=self.app_settings.GENERATION_BACKEND
        )
        self.text_generation_client.set_generation_model(
            model_id=self.app_settings.GENERATION_MODEL_ID
        )

        # Vision client (for image processing)
        self.vision_client = self.llm_provider_factory.create(
            provider=self.app_settings.VISION_BACKEND
        )
        self.vision_client.set_vision_model(
            model_id=self.app_settings.VISION_MODEL_ID
        )

        # Text generation SQL
        self.text_generation_client_sql = self.llm_provider_factory.create(
            provider=self.app_settings.SQL_BACKEND
        )
        self.text_generation_client_sql.set_generation_model(
            model_id=self.app_settings.SQL_MODEL_ID
        )
        self.llm_sql = self.text_generation_client_sql.LLM_CHAT()

        # Initialize SQL_AgentController
        self.sql_agent = SQL_AgentController(self.llm_sql)

        # A custom system prompt describing the ReAct approach

        self.react_system_prompt = self.prompt_template.react_system_prompt()
    #
    # Tools that the ReAct agent can call
    #

    def process_uploaded_image(self, file: UploadFile) -> str:
        """
        Process the uploaded car image to extract details using the vision client.
        """
        try:
            car_details = self.vision_client.vision_to_text(file)
            return car_details
        except Exception as e:
            logging.error(f"Error analyzing the image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error analyzing the image: {str(e)}")

    def handle_sql_mode(self, user_prompt: str) -> str:
        """
        Process SQL-related queries via SQL_AgentController.
        """
        try:
            assistant_response = self.sql_agent.chat_agent_with_sql(user_prompt)
            return assistant_response
        except Exception as e:
            logging.error(f"Error in SQL mode: {str(e)}")
            return f"Error generating SQL response: {str(e)}"



    #
    # The ReAct agent loop
    #

    def react_agent(
        self,
        user_prompt: str,
        conversation_history: str = "",
        car_details: str = ""
    ) -> str:
        """
        Implements the ReAct Agent approach for the user's message.
        We iterate Thought -> Action -> Observation -> Answer. 
        """

        # Keep a local conversation log
        messages = []
        # 1) Add the system prompt (ReAct instructions)
        messages.append({"role": "system", "content": self.react_system_prompt})

        # 2) Optionally add conversation history or extra context
        if conversation_history:
            messages.append({"role": "assistant", "content": f"Conversation history: {conversation_history}"})
        if car_details:
            messages.append({"role": "assistant", "content": f"Car image details: {car_details}"})

        # 3) Add the user's message
        messages.append({"role": "user", "content": user_prompt})

        # We'll limit the loop to avoid infinite steps
        max_iterations = 3

        for _ in range(max_iterations):
            # Invoke the LLM with the current conversation
            assistant_reply = self.text_generation_client.generate_text(
                    prompt=user_prompt,  # Correct prompt type
                    chat_history=messages  # Pass conversation history
                )

            # The content might have Thought, Action, Answer, etc.
            content = assistant_reply 
            # Append to messages (as if it's the model's new response)
            messages.append({"role": "assistant", "content": content})

            # Check for "Answer:"
            if "Answer:" in content:
                # Everything after "Answer:" is the final user answer
                final_answer = content.split("Answer:")[1].strip()
                return final_answer

            # Else, check for "Action:"
            if "Action:" in content:
                # Example: "Action: handle_sql_mode: SELECT something"
                action_part = content.split("Action:")[1].strip()
                if ":" in action_part:
                    tool_name, tool_input = action_part.split(":", 1)
                    tool_name = tool_name.strip()
                    tool_input = tool_input.strip()

                    observation_result = None
                    if tool_name == "handle_sql_mode":
                        observation_result = self.handle_sql_mode(tool_input)

                    elif tool_name == "process_uploaded_image":
                        # In a real scenario you'd call process_uploaded_image(file), 
                        # but here we only have text in "tool_input"
                        observation_result = f"(Mocked) Called process_uploaded_image with: {tool_input}"
                    else:
                        observation_result = f"Unknown tool: {tool_name}"

                    # Append an Observation for the LLM to read
                    messages.append({"role": "system", "content": f"Observation: {observation_result}"})
                    # The loop continues so the LLM can incorporate that observation

                else:
                    messages.append({"role": "system", "content": "Observation: Could not parse Action properly."})
            else:
                # If no Action or Answer is found, it might be just Thought.
                # We continue to let the LLM produce either another Action or an Answer.
                continue

        # If we exhaust the loop without an "Answer:"
        return "I'm sorry, but I couldn't find a final answer."

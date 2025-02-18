import logging
import base64
from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from ..PromptTemplate import get_prompt_template
from openai import OpenAI
import os
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

get_template = get_prompt_template()
class OpenAIProvider(LLMInterface):
    """
    A provider class for using OpenAI models, handling both text and vision tasks.
    Implements the LLMInterface to ensure compatibility and consistency with other LLM backends.
    """

    def __init__(
        self,
        api_key:  str = None,
        base_url: str = None,
        azure_api : str = None ,
        api_version : str = None ,
        azure_endpoint : str = None ,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.0
    ):
        """
        Initializes the OpenAIProvider with default settings and an OpenAI client.

        :param api_key: Your OpenAI API key.
        :param api_url: Optional custom API URL if using a hosted version of OpenAI.
        :param default_input_max_characters: Maximum number of characters allowed in a text prompt.
        :param default_generation_max_output_tokens: Maximum number of tokens to generate in model responses.
        :param default_generation_temperature: Controls randomness in text generation (0.0 = deterministic).
        """
        self.api_key = api_key
        self.azure_api = azure_api
        self.base_url = base_url
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.vision_model_id = None
        self.embedding_model_id = None  

        # Initialize the OpenAI client
        if self.azure_endpoint:
            self.client = AzureOpenAI(
            api_key=self.azure_api ,
            api_version = self.api_version ,
          azure_endpoint=self.azure_endpoint
        )
        else :
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url and len(self.base_url) else None
            )

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str) -> None:
        """
        Sets the model ID to be used for text generation.

        :param model_id: Model identifier for generating text.
        """
        self.generation_model_id = model_id

    def set_vision_model(self, model_id: str) -> None:
        """
        Sets the model ID to be used for vision tasks.

        :param model_id: Model identifier for analyzing images.
        """
        self.vision_model_id = model_id
        
    def set_embedding_model(self, model_id: str):
        """
        Sets the model ID to be used for embedding task.

        :param 
        model_id: Model identifier for text embedding.
        """
        self.embedding_model_id = model_id

    def process_text(self, text: str) -> str:
        """
        Truncates and cleans the input text based on default_input_max_characters.

        :param text: The raw input text.
        :return: A sanitized string within the allowed character limit.
        """
        return text[:self.default_input_max_characters]

    def process_image(self, uploaded_image) -> str:
        """
        Converts an uploaded image into a Base64-encoded string suitable
        for sending to the OpenAI vision models.

        :param uploaded_image: A file-like object representing the uploaded image.
        :return: A Base64-encoded string of the image data.
        """
        #return base64.b64encode(uploaded_image.read()).decode("utf-8")
        uploaded_image.file.seek(0)
        return base64.b64encode(uploaded_image.file.read()).decode("utf-8")
        # return base64.b64encode(uploaded_image.read()).decode("utf-8")

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
        type_chat: str = "agent"
    ) -> str:
        """
        Generates text from the OpenAI model based on the given prompt and chat history.

        :param prompt: User's prompt for text generation.
        :param chat_history: A list of conversation messages for context.
        :param max_output_tokens: Maximum tokens to generate. Defaults to the class default if not provided.
        :param temperature: The temperature for text generation (0.0 = deterministic). Defaults to class default.
        :param type_chat: The type of chat mode (e.g., "agent" or "chat").
        :return: The generated text response, or None if an error occurs.
        """

        if chat_history is None:
            chat_history = []

        # Remove None values from chat_history to avoid errors
        chat_history = [msg for msg in chat_history if msg]

        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None

        if not self.generation_model_id:
            self.logger.error("No generation model has been set for OpenAI.")
            return None

        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature

        # Build messages based on chat type
        if type_chat == "agent":
            messages = [
                {
                    "role": OpenAIEnums.USER.value,
                    "content": prompt,
                },
                *chat_history
            ]
        elif type_chat == "chat":
            messages = [
                {
                    "role": OpenAIEnums.SYSTEM.value,
                    "content": get_template.text_propt_system()
                },
                {
                    "role": OpenAIEnums.USER.value,
                    "content": get_template.text_propt_user(prompt)
                },
                *chat_history
            ]
        else:
            self.logger.error(f"Invalid type_chat: {type_chat}")
            return None  # Ensure we handle unexpected values

        # Debugging: Print messages before sending them to OpenAI
        import json
        print("DEBUG: Sending to OpenAI API:", json.dumps(messages, indent=2))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature
            )
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {str(e)}")
            return None

        # Handle response errors
        if not response or not response.choices or len(response.choices) == 0:
            self.logger.error("Error: Empty response or no choices returned from OpenAI.")
            return None

        if not response.choices[0].message:
            self.logger.error("Error: No message content in the first choice.")
            return None

        return response.choices[0].message.content

    
    def LLM_CHAT(self ,max_output_tokens =None , temperature=None):
        
        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature
        
        
        if self.azure_endpoint:
            os.environ["AZURE_OPENAI_API_KEY"] =  self.azure_api
            os.environ["AZURE_OPENAI_ENDPOINT"] = self.azure_endpoint
            llm_azure =AzureChatOpenAI(
                openai_api_version= self.api_version ,
                azure_deployment=self.generation_model_id,
                model_name=self.generation_model_id,
                max_tokens=max_output_tokens,
                temperature=temperature,
                
                )
            return llm_azure
        else :
            os.environ["OPENAI_API_KEY"] = self.api_key
            llm_openai =ChatOpenAI(
                model=self.generation_model_id ,
                max_tokens=max_output_tokens,
                temperature=temperature,
                )
            return llm_openai
        
    def vision_to_text(self, uploaded_image):
        """
        Sends an image to the OpenAI vision model, returning a textual analysis of the image.

        :param uploaded_image: A file-like object for the image to be analyzed.
        :return: The text result from the vision model, or None on error.
        """
        if not self.client:
            self.logger.error("OpenAI client is not initialized.")
            return None

        if not self.vision_model_id:
            self.logger.error("No vision model has been set for OpenAI.")
            return None

        payload = [
            {
                "role": OpenAIEnums.USER.value,
                "content": [
                    {"type": "text", "text": get_template.get_vision_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self.process_image(uploaded_image)}",
                        },
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.vision_model_id,
            messages=payload
        )

        if not response or not response.choices or len(response.choices) == 0:
            self.logger.error("Error: Empty response or no choices returned from OpenAI vision model.")
            return None

        if not response.choices[0].message:
            self.logger.error("Error: No message content in the vision model's first choice.")
            return None

        return response.choices[0].message.content

    def embed_text(self, text: str):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        response = self.client.embeddings.create(
            model = self.embedding_model_id,
            input = text,
        )

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return response.data[0].embedding

    def construct_prompt(self, prompt: str, role: str) -> dict:
        """
        Constructs a dictionary representing a chat message to be appended to the
        conversation history for OpenAI.

        :param prompt: The text of the message.
        :param role: The role of the message (user, assistant, system).
        :return: A dictionary in the format required by the OpenAI chat API.
        """
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

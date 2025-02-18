import logging
import base64
from ..LLMInterface import LLMInterface
from ..LLMEnums import GroqEnums
from ..PromptTemplate import get_prompt_template
from groq import Groq
from langchain_groq import ChatGroq

get_template = get_prompt_template()

class GroqProvider(LLMInterface):
    """
    A provider class that interfaces with Groq for both text generation and vision tasks.
    Implements the LLMInterface, allowing for customized prompt building, image processing,
    and message handling with the Groq API.
    """

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 10000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.0
    ):
        """
        Initializes the GroqProvider with default settings and a Groq client.

        :param api_key: API key for Groq .
        :param default_input_max_characters: Maximum input size allowed for text prompts.
        :param default_generation_max_output_tokens: Maximum tokens for model output generation.
        :param default_generation_temperature: Temperature for text generation (0.0 = deterministic).
        """
        self.api_key = api_key
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.vision_model_id = None

        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str) -> None:
        """
        Sets the model ID for text generation tasks.
        :param model_id: The ID of the text generation model to use.
        """
        self.generation_model_id = model_id

    def set_vision_model(self, model_id: str) -> None:
        """
        Sets the model ID for vision tasks (image analysis).
        :param model_id: The ID of the vision model to use.
        """
        self.vision_model_id = model_id

    def process_text(self, text: str) -> str:
        """
        Pre-processes and trims the input text according to default input size constraints.
        :param text: The raw text prompt to process.
        :return: A stripped and truncated version of the input text.
        """
        return text[:self.default_input_max_characters].strip()

    def process_image(self, uploaded_image) -> str:
        """
        Encodes the uploaded image file to a Base64 string for sending to Groq.
        :param uploaded_image: An uploaded file-like object containing the image.
        :return: A Base64-encoded string of the image contents.
        # """
        uploaded_image.file.seek(0)
        return base64.b64encode(uploaded_image.file.read()).decode("utf-8")
        # return base64.b64encode(uploaded_image.read()).decode("utf-8")

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None ,
        type_chat :str ="RAG"
    ) -> str:
        """
        Generates text from the model based on the given prompt and optional parameters.
        :param prompt: The user prompt to be sent to the model.
        :param chat_history: A list of previous message objects to provide conversation context.
        :param max_output_tokens: The maximum number of tokens in the generated response.
        :param temperature: The model's sampling temperature (0 = deterministic, higher = more creative).
        :return: The generated response from the Groq model, or None on failure.
        """
        if chat_history is None:
            chat_history = []

        if not self.client:
            self.logger.error("Groq client is not initialized.")
            return None

        if not self.generation_model_id:
            self.logger.error("No generation model has been set for Groq.")
            return None

        # Determine tokens and temperature
        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature

        # Add the new prompt to the chat history
        chat_history.append(self.construct_prompt(prompt=prompt, role=GroqEnums.USER.value))
        if type_chat == "RAG":
                    messages = [
                        {
                            "role": GroqEnums.SYSTEM.value,
                            "content": get_template.rag_system_prompt()
                        },
                        {
                            "role": GroqEnums.USER.value,
                            "content": get_template.rag_user_prompt(prompt)
                        },
                        *chat_history
                    ]
        elif type_chat == "chat" :
            messages = [
                {
                    "role": GroqEnums.SYSTEM.value,
                    "content": get_template.text_propt_system()
                },
                {
                    "role": GroqEnums.USER.value,
                    "content": get_template.text_propt_user(prompt)
                },
                *chat_history
            ]
            


        # Call Groq API for text completion
        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            max_completion_tokens=max_output_tokens,
            temperature=temperature
        )

        # Validate response
        if not response or not response.choices or len(response.choices) == 0:
            self.logger.error("No response or empty choices returned from Groq.")
            return None

        message_content = response.choices[0].message
        if not message_content:
            self.logger.error("Empty message content in the Groq response.")
            return None

        return message_content.content
    
    def LLM_CHAT(self , max_output_tokens =None , temperature =None):
        max_output_tokens = max_output_tokens or self.default_generation_max_output_tokens
        temperature = temperature if temperature is not None else self.default_generation_temperature
        return ChatGroq(
            groq_api_key=self.api_key, 
            model_name=self.generation_model_id  , 
            max_tokens= max_output_tokens ,
            temperature= temperature ,
        
        )

    def vision_to_text(self, uploaded_image):
        """
        Sends an uploaded image to the vision model to extract or describe details about a car.
        :param uploaded_image: An uploaded file-like object representing the image to analyze.
        :return: Text describing the car in the image, or None on failure.
        """
        if not self.client:
            self.logger.error("Groq client is not initialized.")
            return None

        if not self.vision_model_id:
            self.logger.error("No vision model has been set for Groq.")
            return None

        # Prepare the request with the vision prompt and Base64-encoded image
        payload = [
            {
                "role": GroqEnums.USER.value,
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

        # Validate response
        if not response or not response.choices or len(response.choices) == 0:
            self.logger.error("No response or empty choices returned from the Groq vision model.")
            return None

        message_content = response.choices[0].message
        if not message_content:
            self.logger.error("Empty message content in the Groq vision response.")
            return None

        return message_content.content

    def construct_prompt(self, prompt: str, role: str) -> dict:
        """
        Constructs a message dictionary to be appended to the chat history.
        :param prompt: The text of the message (user or assistant).
        :param role: The role of the message (user, assistant, system).
        :return: A dictionary suitable for the Groq chat messages list.
        """
        return {
            "role": role,
            "content": self.process_text(prompt)
        }


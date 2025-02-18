from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, GroqProvider

class LLMProviderFactory:
    def __init__(self, config: dict ,azure =True):
        self.config = config
        self.azure= azure
        
    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            if self.azure :
                return OpenAIProvider(
                    azure_api = self.config.AZURE_OPENAI_API_KEY,
                    api_version = self.config.API_VERSION,
                    azure_endpoint = self.config.AZURE_OPENAI_ENDPOINT,
                    default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                    default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                    default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
                )
            else:
                return OpenAIProvider(
                    api_key = self.config.OPENAI_API_KEY,
                    azure_endpoint =None ,
                    default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                    default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                    default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
                )
        elif provider == LLMEnums.GROQ.value :
            return GroqProvider(
                api_key = self.config.GROQ_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        return None

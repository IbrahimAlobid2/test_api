from abc import ABC, abstractmethod

class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id: str) -> None:
        pass

    @abstractmethod
    def set_vision_model(self, model_id: str) -> None:
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None) -> str:
        pass


    @abstractmethod
    def LLM_CHAT(self):
        pass
    
    
    @abstractmethod
    def vision_to_text(self, uploaded_image, prompt:str):
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str)-> dict:
        pass

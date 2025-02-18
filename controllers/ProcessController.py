from .BaseController import BaseController
import pandas as pd
import os
import logging
from stores.llm.LLMProviderFactory import LLMProviderFactory


class ProcessController(BaseController):

    def __init__(self ):
        super().__init__()


       
        self.llm_provider_factory = LLMProviderFactory(self.app_settings , azure=False)
        self.client = self.llm_provider_factory.create(provider=self.app_settings.EMBEDDING_BACKEND)
        self.client.set_embedding_model(model_id = self.app_settings.EMBEDDING_MODEL_ID)
        self.logger = logging.getLogger(__name__)
        

    def get_file_loader(self, dataset: str):
        
        """
        Load a DataFrame from the specified CSV  file.
        
        Args:
            file_directory (str): The directory path of the file to be loaded.
            
        Returns:
            DataFrame, str: The loaded DataFrame and the file's base name without the extension.
            
        Raises:
            ValueError: If the file extension is neither CSV.
            
        """
        
        file_names_with_extensions = os.path.basename(dataset)
        
        file_name, file_extension = os.path.splitext(
                file_names_with_extensions)
        
        if file_extension == ".csv":
            df = pd.read_csv(dataset)
            return df, file_name
        else:
            self.logger.error("The selected file type is not supported")
            return None
        
    def prepare_data_for_injection(self, df:pd.DataFrame, file_name:str):
        docs = []
        metadatas = []
        ids = []
        embeddings = []
        
        for index, row in df.iterrows():
            output_str = ""
            # Treat each row as a separate chunk
            for col in df.columns:
                output_str += f"{col}: {row[col]},\n"
            print(f'{index} - {output_str}\n')
            embeddings.append(self.client.embed_text( output_str))
            docs.append(output_str)
            metadatas.append({"source": file_name})
            ids.append(f"id{index}")
        return docs, metadatas, ids, embeddings
    
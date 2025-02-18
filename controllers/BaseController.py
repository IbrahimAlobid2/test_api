from helpers.config import get_settings
import os
import re
class BaseController:
    
    def __init__(self):

        self.app_settings = get_settings()
        
        self.base_dir = os.path.dirname( os.path.dirname(__file__) )

        self.database_dir = os.path.join(
            self.base_dir,
            "assets/database"
        )
        
        
    def get_database_path(self, db_name: str):

        database_path = os.path.join(
            self.database_dir, db_name
        )

        if not os.path.exists(database_path):
            os.makedirs(database_path)

        return database_path
    
    def get_dataset_path(self, db_name: str):

        dataset_path = os.path.join(
            self.database_dir, "csv" ,db_name)
        
        return dataset_path
    
    def get_database_sql_path(self, db_name: str):

        database_sql_path = os.path.join(
            self.database_dir, "db_sql" ,db_name)
        
        return database_sql_path
    


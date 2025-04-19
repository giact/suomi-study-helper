from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    app_name: str = 'suomi-study-helper'
    language_code: str = 'fin'
    language_name: str = 'Finnish'
    openai_api_key: str = Field(default=...)

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        pyproject_toml_table_header=('project',),
    )

    def __init__(self):
        # Read the app name from pyproject.toml
        super().__init__()

config = Config()

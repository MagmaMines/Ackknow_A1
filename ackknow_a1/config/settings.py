from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AckknowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACKKNOW_", env_file=".env", extra="ignore")

    app_name: str = "Ackknow A1"
    app_version: str = "2.0.0"
    default_locale: str = "en-IN"
    supported_languages: list[str] = Field(default_factory=lambda: [
        "en-IN", "en-US", "en-UK", "en-AU", "en-NZ", "hi", "hinglish", "awadhi", "kanpuriya", "ta"
    ])

    tokenizer_input_max: int = 500_000
    tokenizer_output_max: int = 230_000

    ollama_base_url: str = "http://localhost:11434"
    ollama_fast_model: str = "qwen2.5:7b-instruct"
    ollama_reasoning_model: str = "qwen2.5:14b-instruct"
    ollama_vision_model: str = "llava:13b"

    sqlite_url: str = "sqlite:///./ackknow.db"
    chroma_path: str = "./chroma_db"

    web_search_enabled: bool = True
    max_web_results: int = 5

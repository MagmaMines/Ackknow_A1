from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AckknowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACKKNOW_", env_file=".env", extra="ignore")

    app_name: str = "Ackknow A1"
    app_version: str = "4.1.0"
    default_locale: str = "en-IN"
    supported_languages: list[str] = Field(default_factory=lambda: [
        "en-IN", "en-US", "en-UK", "en-AU", "en-NZ", "hi", "hinglish", "awadhi", "kanpuriya", "ta"
    ])

    llm_backend: str = "own"
    own_model_path: str = "./models/ackknow_a1_v4"
    own_model_device: str = "cpu"
    own_model_max_input_tokens: int = 100_000
    own_model_max_new_tokens: int = 25_000
    own_model_temperature: float = 0.65
    own_model_target_params_b: float = 40.0
    own_model_compile: bool = True
    own_model_quantization: str = "int8_dynamic"

    ollama_base_url: str = "http://localhost:11434"
    ollama_fast_model: str = "qwen2.5:7b-instruct"
    ollama_reasoning_model: str = "qwen2.5:14b-instruct"
    ollama_vision_model: str = "llava:13b"

    sqlite_url: str = "sqlite:///./ackknow.db"
    web_search_enabled: bool = True
    max_web_results: int = 5

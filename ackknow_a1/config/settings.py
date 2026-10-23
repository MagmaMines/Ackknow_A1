from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AckknowSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ACKKNOW_", env_file=".env", extra="ignore")

    app_name: str = "Ackknow A1"
    app_version: str = "5.0.0"
    default_locale: str = "en-IN"
    supported_languages: list[str] = Field(default_factory=lambda: [
        "en-IN", "en-US", "en-UK", "en-AU", "en-NZ", "hi", "hinglish", "awadhi", "kanpuriya", "ta"
    ])

    llm_backend: str = "own"
    own_model_path: str = "./models/ackknow_a1_v5"
    own_model_checkpoint: str = "model.pt"
    own_model_device: str = "cpu"
    own_model_max_input_tokens: int = 100_000
    own_model_max_new_tokens: int = 25_000
    own_model_temperature: float = 0.65
    own_model_compile: bool = True
    own_model_quantization: str = "int8_dynamic"

    termux_mode: bool = False
    mobile_profile: str = "balanced"  # light|balanced|max

    ollama_base_url: str = "http://localhost:11434"
    ollama_fast_model: str = "qwen2.5:7b-instruct"
    ollama_reasoning_model: str = "qwen2.5:14b-instruct"
    ollama_vision_model: str = "llava:13b"
    ollama_code_model: str = "qwen2.5-coder:7b-instruct"
    llm_fallback_enabled: bool = True

    code_temperature: float = 0.2
    code_verify_enabled: bool = True
    code_exec_enabled: bool = False
    code_exec_timeout_s: int = 10
    code_max_repair_rounds: int = 2

    sqlite_url: str = "sqlite:///./ackknow.db"
    web_search_enabled: bool = True
    max_web_results: int = 5

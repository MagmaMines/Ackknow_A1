from dataclasses import dataclass, field


@dataclass
class TokenizerSettings:
    input_tokens_max: int = 500_000
    output_tokens_max: int = 230_000


@dataclass
class LanguageSettings:
    supported: tuple[str, ...] = (
        "en-IN", "en-UK", "en-US", "en-AU", "en-NZ",
        "hi", "hinglish", "ta", "awadhi", "kanpuriya",
    )
    default: str = "en-US"


@dataclass
class AckknowSettings:
    model_name: str = "Ackknow A1"
    tokenizer: TokenizerSettings = field(default_factory=TokenizerSettings)
    languages: LanguageSettings = field(default_factory=LanguageSettings)
    predata_topics: tuple[str, ...] = (
        "MagmaMinesTeam", "Apple Inc", "Samsung", "Microsoft",
        "Banana", "India", "USA", "China",
    )

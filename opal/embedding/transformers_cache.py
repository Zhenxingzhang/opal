import os
import typing

DEFAULT_SENTENCE_TRANSFORMERS_CACHE_PATH: typing.Final[str] = os.path.expanduser(
    "~/.sentence_transformers"
)


def set_sentence_transformers_cache_env_var() -> None:
    if os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
        return  # < explicitly set so we do not override it

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = DEFAULT_SENTENCE_TRANSFORMERS_CACHE_PATH


def get_sentence_transformers_cache_dir() -> str:
    return os.getenv(
        "SENTENCE_TRANSFORMERS_HOME",
        DEFAULT_SENTENCE_TRANSFORMERS_CACHE_PATH,
    )


def set_huggingface_cache_env_var() -> None:
    """Set the cache directory for sentence transformers and HuggingFace."""
    if os.environ.get("HF_HUB_CACHE"):
        return  # < explicitly set so we do not override it

    os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.hf_cache")

"""Centralized caching for embedding models (SentenceTransformer and CrossEncoder) across the codebase."""

import fcntl
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Generic, TypeVar

import requests
from huggingface_hub.errors import HfHubHTTPError
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_agents.embedding.transformers_cache import get_sentence_transformers_cache_dir

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ProcessSafeModelLoader(Generic[T]):
    """Generic process-safe loader for ML models with file-based synchronization.

    Each model gets its own subfolder containing:
    - .lock file for process synchronization
    - .downloaded marker file to indicate completion
    - The actual model files downloaded by the library
    """

    def __init__(
        self,
        model_type: str,
        cache_folder: Path,
        loader_func: Callable[[str, str, bool], T],
    ):
        """Initialize the loader.

        Args:
            model_type: Type identifier for the model (e.g., "sentence_transformer")
            cache_folder: Base cache folder for models
            loader_func: Function to load/download the model (name, cache_folder, local_files_only)
        """
        self.model_type = model_type
        self.cache_folder = cache_folder
        self.loader_func = loader_func

    def load(self, model_name: str) -> T:
        """Load a model, downloading if necessary with process synchronization.

        Args:
            model_name: Name/identifier of the model to load

        Returns:
            Loaded model instance
        """
        # Get the dedicated folder for this model
        model_folder = self._get_model_folder(model_name)
        model_folder.mkdir(parents=True, exist_ok=True)

        downloaded_marker = model_folder / ".downloaded"
        lock_file_path = model_folder / ".lock"

        # Fast path: if already downloaded, just load it from local files
        if downloaded_marker.exists():
            logger.debug(
                f"Loading cached {self.model_type} {model_name} from local files (pid={os.getpid()})"
            )
            return self.loader_func(model_name, str(self.cache_folder), True)

        # Slow path: need to coordinate download
        with open(lock_file_path, "a+") as lock_file:
            timeout = 300  # 5 minutes
            start = time.time()

            while True:
                # Check if model was downloaded while waiting
                if downloaded_marker.exists():
                    logger.debug(
                        f"Model {model_name} appeared, loading from local files (pid={os.getpid()})"
                    )
                    return self.loader_func(model_name, str(self.cache_folder), True)

                try:
                    # Try to acquire exclusive lock
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Double-check after acquiring lock
                    if downloaded_marker.exists():
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                        return self.loader_func(
                            model_name, str(self.cache_folder), True
                        )

                    # We have the lock and model isn't downloaded - download it
                    logger.info(
                        f"Downloading {self.model_type} {model_name} (pid={os.getpid()})"
                    )

                    try:
                        # Download from network (local_files_only=False)
                        model = self.loader_func(
                            model_name, str(self.cache_folder), False
                        )
                        # Mark as downloaded
                        downloaded_marker.touch()
                        logger.info(f"Downloaded {model_name} successfully")
                    except Exception as e:
                        logger.error(f"Failed to download {model_name}: {e}")
                        raise
                    finally:
                        # Always release the lock
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

                    return model

                except BlockingIOError:
                    # Lock held by another process, wait and retry
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Timeout waiting for {model_name} download")
                    time.sleep(0.1)

    def _get_model_folder(self, model_name: str) -> Path:
        """Get the dedicated folder for a model.

        Creates a safe folder name from the model name.
        """
        # Create a safe folder name by replacing special characters
        safe_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_folder / "sync" / f"{self.model_type}_{safe_name}"


RETRYABLE_EXCEPTIONS = (
    # Network errors
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ChunkedEncodingError,
    # Hugging Face specific error
    HfHubHTTPError,
)


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _load_sentence_transformer_with_retries(
    name: str, hf_cache_folder: str | Path, local: bool
) -> SentenceTransformer:
    """Load sentence transformer model with retries.

    Uses HuggingFace's built-in caching and file locking for cross-process coordination.
    """
    return SentenceTransformer(
        name,
        cache_folder=str(hf_cache_folder),
        local_files_only=local,
        trust_remote_code=False,
    )


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _load_cross_encoder_with_retries(
    name: str, hf_cache_folder: str | Path, local: bool
) -> CrossEncoder:
    """Load cross encoder model with retries.

    Uses HuggingFace's built-in caching and file locking for cross-process coordination.
    """
    return CrossEncoder(  # type: ignore[no-any-return]
        name,
        cache_folder=str(hf_cache_folder),
        local_files_only=local,
        trust_remote_code=False,
    )


class EmbeddingModelCache:
    """Thread-safe singleton cache for SentenceTransformer and CrossEncoder model instances."""

    _sentence_transformers: dict[str, SentenceTransformer] = {}
    _cross_encoders: dict[str, CrossEncoder] = {}
    _lock = threading.Lock()

    # Lazy-initialized loaders
    _st_loader: ProcessSafeModelLoader[SentenceTransformer] | None = None
    _ce_loader: ProcessSafeModelLoader[CrossEncoder] | None = None

    @classmethod
    def _get_st_loader(cls) -> ProcessSafeModelLoader[SentenceTransformer]:
        """Get or create the SentenceTransformer loader."""
        if cls._st_loader is None:
            cache_folder = Path(cls.cache_folder())
            cls._st_loader = ProcessSafeModelLoader(
                model_type="sentence_transformer",
                cache_folder=cache_folder,
                loader_func=_load_sentence_transformer_with_retries,
            )
        return cls._st_loader

    @classmethod
    def _get_ce_loader(cls) -> ProcessSafeModelLoader[CrossEncoder]:
        """Get or create the CrossEncoder loader."""
        if cls._ce_loader is None:
            cache_folder = Path(cls.cache_folder())
            cls._ce_loader = ProcessSafeModelLoader(
                model_type="cross_encoder",
                cache_folder=cache_folder,
                loader_func=_load_cross_encoder_with_retries,
            )
        return cls._ce_loader

    @classmethod
    def get_sentence_transformer(
        cls, model_name: str = "all-MiniLM-L6-v2"
    ) -> SentenceTransformer:
        """Get or create a cached SentenceTransformer model instance.

        This method is thread-safe and process-safe, ensuring proper synchronization
        across parallel processes (e.g., pytest with -n auto).

        Args:
            model_name: Name of the model to load. Defaults to "all-MiniLM-L6-v2".

        Returns:
            Cached SentenceTransformer instance
        """
        with cls._lock:
            # Check process-local cache first
            if model_name in cls._sentence_transformers:
                return cls._sentence_transformers[model_name]
            # Load model (with process synchronization if needed)
            logger.info(f"Loading embedding model {model_name} ...")
            model = cls._get_st_loader().load(model_name)

            # Cache in process memory
            cls._sentence_transformers[model_name] = model
            return model

    @classmethod
    def get_cross_encoder(
        cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> CrossEncoder:
        """Get or create a cached CrossEncoder model instance.

        This method is thread-safe and process-safe, ensuring proper synchronization
        across parallel processes (e.g., pytest with -n auto).

        Args:
            model_name: Name of the model to load.

        Returns:
            Cached CrossEncoder instance
        """
        # Check and load under lock to prevent race conditions
        with cls._lock:
            # Check process-local cache first
            if model_name in cls._cross_encoders:
                return cls._cross_encoders[model_name]

            # Load model (with process synchronization if needed)
            model = cls._get_ce_loader().load(model_name)

            # Cache in process memory
            cls._cross_encoders[model_name] = model
            return model  # type: ignore[no-any-return]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models. Useful for testing or memory management."""
        with cls._lock:
            cls._sentence_transformers.clear()
            cls._cross_encoders.clear()
            cls._st_loader = None
            cls._ce_loader = None

    @classmethod
    def clear_cache_folder(cls) -> None:
        """Clear the cache folder on disk."""

        cache_folder = cls.cache_folder()
        if os.path.exists(cache_folder):
            logger.debug(f"Clearing {cache_folder}")
            shutil.rmtree(cache_folder, ignore_errors=True)

        hf_cache = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        if os.path.exists(hf_cache):
            logger.debug(f"Clearing {hf_cache}")
            shutil.rmtree(hf_cache, ignore_errors=True)

    @classmethod
    def cache_folder(cls) -> str:
        """Get the cache folder path."""
        return get_sentence_transformers_cache_dir()


# Convenience functions for direct access
def get_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """Get a cached SentenceTransformer model instance.

    Args:
        model_name: Name of the model to load. Defaults to "all-MiniLM-L6-v2".

    Returns:
        Cached SentenceTransformer instance
    """
    return EmbeddingModelCache.get_sentence_transformer(model_name)


def get_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoder:
    """Get a cached CrossEncoder model instance.

    Args:
        model_name: Name of the model to load.

    Returns:
        Cached CrossEncoder instance
    """
    return EmbeddingModelCache.get_cross_encoder(model_name)

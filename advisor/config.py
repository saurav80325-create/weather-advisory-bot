import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "policies"


def get_llm():
    """Model is chosen by env var, e.g. LLM_MODEL=openai:gpt-4o-mini. Temperature 0 for repeatability."""
    from langchain.chat_models import init_chat_model
    return init_chat_model(os.getenv("LLM_MODEL", "anthropic:claude-haiku-4-5-20251001"), temperature=0, max_retries=6)

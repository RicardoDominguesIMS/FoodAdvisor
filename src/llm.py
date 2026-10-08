"""Single place where the LLM client is created. Reads settings from the environment."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # loads .env if present; harmless if not

# Course proxy defaults (not secrets). Only LLM_API_KEY is personal to your group.
DEFAULT_BASE_URL = "https://llm-proxy-obfzuamujq-ew.a.run.app/v1"
DEFAULT_MODEL = "course-chat"


def get_client() -> OpenAI:
    base_url = os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set. Check your .env file.")
    return OpenAI(base_url=base_url, api_key=api_key)


def ask(prompt: str, system: str | None = None) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    response = get_client().chat.completions.create(
        model=os.environ.get("LLM_MODEL") or DEFAULT_MODEL, messages=messages
    )
    return response.choices[0].message.content

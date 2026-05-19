import os
from openai import OpenAI
from .base import AIBackend


class OpenAIBackend(AIBackend):
    def __init__(self, config: dict):
        api_key = os.environ.get(config["ai"]["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {config['ai']['api_key_env']} is not set."
            )
        self.client = OpenAI(api_key=api_key)

    def complete(self, messages: list[dict], config: dict) -> tuple[str, int]:
        ai_cfg = config["ai"]
        resp = self.client.chat.completions.create(
            model=ai_cfg["model"],
            messages=messages,
            max_tokens=ai_cfg["max_tokens_response"],
            temperature=ai_cfg["temperature"],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
        tokens = resp.usage.total_tokens
        return text, tokens

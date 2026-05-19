import os
import anthropic
from .base import AIBackend


class AnthropicBackend(AIBackend):
    def __init__(self, config: dict):
        api_key = os.environ.get(config["ai"]["api_key_env"])
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {config['ai']['api_key_env']} is not set."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages: list[dict], config: dict) -> tuple[str, int]:
        ai_cfg = config["ai"]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]

        response = self.client.messages.create(
            model=ai_cfg["model"],
            max_tokens=ai_cfg["max_tokens_response"],
            temperature=ai_cfg["temperature"],
            system=system,
            messages=user_msgs,
        )
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens

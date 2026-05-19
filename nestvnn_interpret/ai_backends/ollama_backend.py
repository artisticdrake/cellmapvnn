import requests
from .base import AIBackend


class OllamaBackend(AIBackend):
    def __init__(self, config: dict):
        self.base_url = config["ai"].get("ollama_base_url", "http://localhost:11434")

    def complete(self, messages: list[dict], config: dict) -> tuple[str, int]:
        ai_cfg = config["ai"]
        payload = {
            "model": ai_cfg["model"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": ai_cfg["temperature"]},
            "format": "json",
        }
        resp = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=300
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["message"]["content"]
        tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
        return text, tokens

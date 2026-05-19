from abc import ABC, abstractmethod


class AIBackend(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], config: dict) -> tuple[str, int]:
        """
        Send messages to the AI.
        Returns (response_text, tokens_used).
        response_text must be the raw string from the model — do not parse here.
        """

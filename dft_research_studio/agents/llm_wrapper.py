"""
agents/llm_wrapper.py
----------------------
Ollama wrapper with Groq fallback, token telemetry and cost estimation.
"""
from __future__ import annotations
import os, time, requests
from typing import Tuple
from ..config import Config

_OLLAMA_URL = "http://localhost:11434/api/chat"

class LLMWrapper:
    def __init__(self, model_name: str, config: Config | None = None) -> None:
        cfg = config or Config()
        self.model_name = model_name
        self.use_ollama = False
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                self.use_ollama = True
        except Exception:
            pass
        if not self.use_ollama:
            from groq import Groq
            self.client = Groq(api_key=cfg.groq_api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> Tuple[str, int, int, int, float, int]:
        if self.use_ollama:
            return self._generate_ollama(prompt, max_tokens, temperature)
        return self._generate_groq(prompt, max_tokens, temperature)

    def _generate_ollama(self, prompt, max_tokens, temperature):
        for attempt in range(3):
            try:
                resp = requests.post(_OLLAMA_URL, json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    }
                }, timeout=300)
                data = resp.json()
                text = data.get("message", {}).get("content", "")
                it = data.get("prompt_eval_count", 0)
                ot = data.get("eval_count", 0)
                tt = it + ot
                if text.strip():
                    return text, it, ot, tt, 0.0, 1
                print(f"[LLMWrapper] Empty response from {self.model_name}, retry {attempt+1}")
                time.sleep(2)
            except Exception as exc:
                print(f"[LLMWrapper] Ollama error ({self.model_name}, attempt {attempt+1}): {exc}")
                time.sleep(3)
        return "", 0, 0, 0, 0.0, 0

    def _generate_groq(self, prompt, max_tokens, temperature):
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                it = resp.usage.prompt_tokens
                ot = resp.usage.completion_tokens
                tt = it + ot
                text = resp.choices[0].message.content or ""
                if text.strip():
                    return text, it, ot, tt, 0.0, 1
                time.sleep(3 * (attempt + 1))
            except Exception as exc:
                print(f"[LLMWrapper] Groq error ({self.model_name}, attempt {attempt+1}): {exc}")
                time.sleep(5 * (attempt + 1))
        return "", 0, 0, 0, 0.0, 0

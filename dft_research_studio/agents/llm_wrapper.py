"""
agents/llm_wrapper.py
----------------------
Thin Groq SDK wrapper with token telemetry and cost estimation.
"""

from __future__ import annotations

import os
from typing import Tuple

from groq import Groq

from ..config import Config


# Cost table (USD per 1 M tokens) — verify current prices at console.groq.com
_PRICING: dict[str, tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.20, 0.70),
}


class LLMWrapper:
    """
    Wraps the Groq client and returns a 6-tuple on every generation call:
    (response_text, input_tokens, output_tokens, total_tokens, cost_usd, api_calls)
    """

    def __init__(self, model_name: str, config: Config | None = None) -> None:
        cfg = config or Config()
        self.model_name = model_name
        self.client = Groq(api_key=cfg.groq_api_key)

        in_cost, out_cost = _PRICING.get(model_name, (0.0, 0.0))
        self._in_cost = in_cost
        self._out_cost = out_cost

    def generate(
        self,
        prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> Tuple[str, int, int, int, float, int]:
        """
        Returns
        -------
        (text, input_tokens, output_tokens, total_tokens, cost_usd, api_calls)
        """
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
            cost = (
                it / 1_000_000 * self._in_cost
                + ot / 1_000_000 * self._out_cost
            )
            return resp.choices[0].message.content, it, ot, tt, cost, 1
        except Exception as exc:
            print(f"[LLMWrapper] Error calling {self.model_name}: {exc}")
            return "", 0, 0, 0, 0.0, 0

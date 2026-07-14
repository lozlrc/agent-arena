"""OpenAI-backed Prisoner's Dilemma agent for the `dilemma_comms` game.

This agent is *trusted* code: it makes network calls, so it is run from
the registry / CLI, never through the untrusted `/api/evaluate` path
(whose AST sandbox blocks imports and network on purpose). The openai
SDK is imported lazily, so this module is importable — and the whole
test suite runs — without openai installed or any API key set.

Configuration (constructor args override environment):
  OPENAI_API_KEY   required at call time
  OPENAI_MODEL     model id (default: gpt-4o-mini) — set to whatever you
                   have access to
Determinism: OpenAI sampling is best-effort only. We pass a fixed
`seed` and low-ish temperature, but identical replies are not
guaranteed, so comms matches are statistically (not byte-)reproducible.
"""

from __future__ import annotations

import os

from arena.game.dilemma import COOPERATE, DEFECT, MAX_MSG_LEN, PDAgent

DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are playing an iterated Prisoner's Dilemma against one opponent. "
    "Each round you first exchange a short free-text message, then secretly "
    "choose COOPERATE or DEFECT. Payoffs per round (yours/theirs): both "
    "cooperate 3/3; you defect while they cooperate 5/0; you cooperate while "
    "they defect 0/5; both defect 1/1. There is a 2% chance any move is "
    "flipped by noise. Your sole goal is to maximize YOUR OWN total score "
    "over the game. Messages are cheap talk: you may say anything, and so "
    "may your opponent — neither of you is bound by what you say."
)


def _history_str(obs: dict) -> str:
    hs, ho = obs["history_self"], obs["history_opp"]
    if not hs:
        return "No rounds played yet."
    def c(x):
        return "C" if x else "D"
    rounds = ", ".join(f"r{i}: you={c(a)} them={c(b)}"
                       for i, (a, b) in enumerate(zip(hs, ho)))
    return (f"History ({len(hs)} rounds) — {rounds}. "
            f"Score so far: you={obs['score_self']} them={obs['score_opp']}.")


def _messages_str(obs: dict) -> str:
    msgs = obs.get("messages_this_round", ())
    if not msgs:
        return "No messages this round yet."
    who = {obs["player_id"]: "you", 1 - obs["player_id"]: "opponent"}
    return " ".join(f'{who[m["seat"]]}: "{m["text"]}"' for m in msgs)


class OpenAIPDAgent(PDAgent):
    """PD agent whose message() and play() are produced by an OpenAI model."""

    name = "llm_openai"

    def __init__(self, model: str | None = None, temperature: float = 0.7,
                 api_key: str | None = None, seed: int = 7,
                 name: str | None = None):
        self._model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self._temperature = temperature
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._seed = seed
        self._client = None
        if name:
            self.name = name

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "openai SDK not installed — run `uv sync --extra llm`") from e
            if not self._api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set (env var or constructor arg)")
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _chat(self, prompt: str, max_tokens: int) -> str:
        resp = self._get_client().chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=max_tokens,
            seed=self._seed,
        )
        return (resp.choices[0].message.content or "").strip()

    def message(self, obs: dict) -> str:
        prompt = (
            f"{_history_str(obs)}\n{_messages_str(obs)}\n\n"
            f"Send a brief message (<= {MAX_MSG_LEN} chars) to your opponent "
            "before this round's move. Reply with only the message text.")
        return self._chat(prompt, max_tokens=80)[:MAX_MSG_LEN]

    def play(self, obs: dict) -> bool:
        prompt = (
            f"{_history_str(obs)}\n"
            f"Messages this round — {_messages_str(obs)}\n\n"
            "Choose your move for this round. Reply with exactly one word: "
            "COOPERATE or DEFECT.")
        reply = self._chat(prompt, max_tokens=4).lower()
        # Parse robustly; default to COOPERATE if the model is ambiguous.
        if "defect" in reply and "cooperate" not in reply:
            return DEFECT
        return COOPERATE

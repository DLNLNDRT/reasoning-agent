from dataclasses import dataclass
from typing import List, Optional, Iterator
import os
import time, random


@dataclass
class GenOut:
    texts: List[str]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

class LLM:
    def generate(self, prompt: str, temperature: float = 0.0, n: int = 1) -> GenOut:
        raise NotImplementedError

    # Optional: backends may implement this
    def generate_stream(self, prompt: str, temperature: float = 0.0) -> Iterator[str]:
        # By default, fall back to non-streaming generate()
        out = self.generate(prompt, temperature=temperature, n=1)
        text = out.texts[0] if out.texts else ""
        if text:
            yield text


# ---------- Frontier: OpenAI (ChatGPT family) ----------
class OpenAIChat(LLM):
    def __init__(self, model=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("FRONTIER_MODEL", "gpt-4o")

    def _supports_temperature(self) -> bool:
        # GPT-5 doesn’t allow temperature
        return not self.model.startswith("gpt-5")

    def generate_stream(self, prompt: str, temperature: float = 0.0,  system: str | None = "You are a careful expert test-taker."):
        """
        Yield string chunks from OpenAI Chat Completions streaming API.
        """
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        # Build kwargs and only include temperature if supported
        kwargs = dict(model=self.model, messages=msgs, stream=True)
        if self._supports_temperature():
            kwargs["temperature"] = float(temperature)

        try:
            resp = self.client.chat.completions.create(**kwargs)
            for chunk in resp:
                # v1 API: choices[0].delta.content may be None
                delta = getattr(chunk.choices[0], "delta", None)
                text = getattr(delta, "content", None)
                if isinstance(text, str) and text:
                    yield text
        except (StopIteration, RuntimeError):
            # Gracefully end stream
            pass
        except Exception:
            # Fallback to one-shot generate if streaming fails
            fallback = self.generate(prompt, temperature=temperature).texts[0]
            if fallback:
                yield fallback

    def generate(self, prompt: str, temperature: float = 0.0, n: int = 1, system: str | None = "You are a careful expert test-taker.") -> GenOut:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        # Build kwargs and only include temperature if supported
        kwargs = dict(model=self.model, messages=msgs, n=n)
        if self._supports_temperature():
            kwargs["temperature"] = float(temperature)

        resp = self.client.chat.completions.create(**kwargs)
        texts = [c.message.content or "" for c in resp.choices]
        return GenOut(texts=texts)


# ---------- Frontier: Google Gemini ----------
class GeminiChat(LLM):
    def __init__(self, model=None):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)

        self.model_name = model or os.getenv("GOOGLE_MODEL", "gemini-2.5-pro")

        from google.generativeai import GenerativeModel
        self.model = GenerativeModel(
            model_name=self.model_name,
            system_instruction="Answer concisely. For multiple choice, return only A/B/C/D.",
        )
        self.model_no_sys = GenerativeModel(model_name=self.model_name)  # no system_instruction

        # Version-compatible safety settings: typed (newer SDK) → dicts (older SDK)
        try:
            from google.generativeai.types import SafetySetting, HarmCategory, HarmBlockThreshold
            self.safety_settings = [
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
                SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH),
            ]
        except Exception:
            # Older SDKs accept simple dicts
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT",         "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_ONLY_HIGH"},
            ]


    # --- unified retry helper for transient Google API errors ---
    def _call_with_retry(self, fn, *, max_retries=4, base_sleep=3.0):
        """
        Call a function, retrying on transient Google API errors (429/500/503).
        Uses server-provided retry delay if present, else exponential backoff with jitter.
        """
        import google.api_core.exceptions as gexc
        sleep = base_sleep
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except (gexc.ResourceExhausted, gexc.InternalServerError, gexc.ServiceUnavailable) as e:
                last_err = e
                # server hint for 429 if present
                delay = getattr(getattr(e, "retry_delay", None), "seconds", None)
                if delay is None:
                    delay = int(sleep + random.uniform(0, 1.5))
                    sleep *= 1.6  # backoff
                print(f"[Gemini retry] {e.__class__.__name__}: sleeping {delay}s (attempt {attempt}/{max_retries})")
                time.sleep(delay)
        # ran out of retries
        raise last_err

    # ---- helpers ----
    def _coalesce_candidate_text(self, candidate) -> str:
        """Join all text parts from a candidate; return '' if none."""
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return ""
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str) and t:
                texts.append(t)
        return "\n".join(texts).strip()

    def _extract_first_text(self, resp) -> str:
        """Safe extraction that never uses resp.text quick accessor."""
        fb = getattr(resp, "prompt_feedback", None)
        if fb and getattr(fb, "block_reason", None):
            return f"[Gemini blocked prompt: {fb.block_reason}]"
        for cand in getattr(resp, "candidates", []) or []:
            text = self._coalesce_candidate_text(cand)
            if text:
                return text
        fr = None
        if getattr(resp, "candidates", None):
            fr = getattr(resp.candidates[0], "finish_reason", None)
        return f"[Gemini returned no text parts; finish_reason={fr}]"

    # ---- non-streaming ----
    def generate(self, prompt: str, temperature: float = 0.0, n: int = 1, system: str | None = "default") -> GenOut:
        texts = []
        use_model = self.model if system not in (None, "") else self.model_no_sys
        for _ in range(n):
            def _once():
                return use_model.generate_content(
                    prompt,
                    generation_config={"temperature": float(temperature), "max_output_tokens": 512},
                    safety_settings=self.safety_settings,
                )
            resp = self._call_with_retry(_once)
            texts.append(self._extract_first_text(resp))
        return GenOut(texts=texts)

    # ---- streaming ----
    def generate_stream(self, prompt: str, temperature: float = 0.0, system: str | None = "default"):
        use_model = self.model if system not in (None, "") else self.model_no_sys
        """
        Yield text chunks from Gemini streaming; if chunks have no text parts, skip them.
        On failure, fall back to a single non-streaming response.
        """
        def _start_stream():
            return use_model.generate_content(
                prompt,
                generation_config={"temperature": float(temperature)},
                safety_settings=self.safety_settings,
                stream=True,
            )

        # Start stream with retries; if we can't, fallback to one-shot
        try:
            resp = self._call_with_retry(_start_stream)
        except Exception:
            fallback = self.generate(prompt, temperature=temperature, system=system).texts[0]
            if fallback:
                yield fallback
            return

        try:
            for chunk in resp:
                cands = getattr(chunk, "candidates", None)
                if not cands:
                    continue
                content = getattr(cands[0], "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                for p in parts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t:
                        yield t
        except Exception:
            # Fallback to one-shot on any streaming error
            fallback = self.generate(prompt, temperature=temperature, system=system).texts[0]
            if fallback:
                yield fallback

# ---------- Small: Ollama (local models) ----------
class OllamaChat(LLM):
    def __init__(self, model=None):
        import ollama
        self.ollama = ollama
        self.model = model or os.getenv("SMALL_MODEL", "gemma2:9b")

    def generate_stream(self, prompt: str, temperature: float = 0.0, system: str | None = "default"):
        """
        Yield text chunks from Ollama streaming API (chat with stream=True).
        """
        try:
            stream = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": float(temperature)},
                stream=True,
            )
            for chunk in stream:
                    msg = chunk.get("message", {})
                    t = msg.get("content", "")
                    if t:
                        yield t
        except (StopIteration, RuntimeError):
            pass
        except Exception:
            # Fallback to one-shot generate
            fallback = self.generate(prompt, temperature=temperature).texts[0]
            if fallback:
                yield fallback

    def generate(self, prompt: str, temperature: float = 0.0, n: int = 1, system: str | None = "default") -> GenOut:
        texts = []
        for _ in range(n):
            out = self.ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": float(temperature)},
            )
            texts.append(out["message"]["content"])
        return GenOut(texts=texts)


def build_llm(provider: str, model: str) -> LLM:
    if provider == "openai": return OpenAIChat(model=model)
    if provider == "google": return GeminiChat(model=model)
    if provider == "ollama": return OllamaChat(model=model)
    raise ValueError(f"Unknown provider: {provider}")

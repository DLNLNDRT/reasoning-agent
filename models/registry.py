from dataclasses import dataclass
from typing import List, Optional
from typing import Iterator
import os

@dataclass
class GenOut:
    texts: List[str]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

class LLM:
    def generate(self, prompt: str, temperature=0.0, n=1) -> GenOut:
        raise NotImplementedError

# ---------- Frontier: OpenAI (ChatGPT family) ----------
class OpenAIChat(LLM):
    def __init__(self, model=None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("FRONTIER_MODEL", "gpt-4o")
        
    def generate_stream(self, prompt, temperature=0.0) -> Iterator[str]:
        # Streaming chat completions
        msgs = [
            {"role": "system", "content": "You are a careful expert test-taker."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs,
            temperature=float(temperature), stream=True
        )
        try:
            for chunk in resp:
                # v1 API: choices[0].delta.content may be None
                delta = getattr(chunk.choices[0], "delta", None)
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except StopIteration:
            return
        except Exception:
            # swallow streaming glitch; higher layer will fallback
            return

    def generate(self, prompt, temperature=0.0, n=1) -> GenOut:
        msgs = [
            {"role": "system", "content": "You are a careful expert test-taker."},
            {"role": "user", "content": prompt},
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=float(temperature), n=n
        )

        texts = [c.message.content for c in resp.choices]
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

        # Optional: relax safety if harmless prompts get blocked. Uncomment to use.
        # from google.generativeai.types import SafetySetting
        # self.safety_settings = {
        #     "HARASSMENT": SafetySetting(block_threshold="BLOCK_ONLY_HIGH"),
        #     "HATE_SPEECH": SafetySetting(block_threshold="BLOCK_ONLY_HIGH"),
        #     "SEXUAL_CONTENT": SafetySetting(block_threshold="BLOCK_ONLY_HIGH"),
        #     "DANGEROUS_CONTENT": SafetySetting(block_threshold="BLOCK_ONLY_HIGH"),
        # }
        self.safety_settings = None

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
            if t:
                texts.append(t)
        return "\n".join(texts).strip()

    def _extract_first_text(self, resp) -> str:
        """Safe extraction that never uses resp.text quick accessor."""
        # Check prompt-level block
        fb = getattr(resp, "prompt_feedback", None)
        if fb and getattr(fb, "block_reason", None):
            return f"[Gemini blocked prompt: {fb.block_reason}]"

        # Walk candidates and pull any text parts
        for cand in getattr(resp, "candidates", []) or []:
            text = self._coalesce_candidate_text(cand)
            if text:
                return text

        # Nothing found — include finish_reason for debugging if available
        fr = None
        if getattr(resp, "candidates", None):
            fr = getattr(resp.candidates[0], "finish_reason", None)
        return f"[Gemini returned no text parts; finish_reason={fr}]"

    # ---- non-streaming ----
    def generate(self, prompt, temperature=0.0, n=1) -> GenOut:
        texts = []
        for _ in range(n):
            resp = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": float(temperature),
                    "max_output_tokens": 512,  # keep modest to avoid truncation surprises
                },
                safety_settings=self.safety_settings,
            )
            texts.append(self._extract_first_text(resp))
        return GenOut(texts=texts)

    # ---- streaming ----
    def generate_stream(self, prompt, temperature=0.0) -> Iterator[str]:
        """Yield text chunks; if chunks have no text parts, silently skip them."""
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": float(temperature)},
            safety_settings=self.safety_settings,
            stream=True,
        )
        try:
            for chunk in resp:
                # Don’t touch chunk.text; pull parts safely
                cands = getattr(chunk, "candidates", None)
                if not cands:
                    continue
                parts = getattr(cands[0].content, "parts", None)
                if not parts:
                    continue
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        yield t
        except Exception:
            # Fallback to one-shot if streaming fails mid-way
            fallback = self.generate(prompt, temperature=temperature).texts[0]
            if fallback:
                yield fallback

        return GenOut(texts=texts)


# ---------- Small: Ollama (local models) ----------
class OllamaChat(LLM):
    def __init__(self, model=None):
        import ollama
        self.ollama = ollama
        self.model = model or os.getenv("SMALL_MODEL", "gemma2:9b")

    def generate_stream(self, prompt, temperature=0.0) -> Iterator[str]:
        # Ollama Python SDK stream=True returns an iterator of chunks
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
        except StopIteration:
            return
        except Exception:
            return

    def generate(self, prompt, temperature=0.0, n=1) -> GenOut:
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

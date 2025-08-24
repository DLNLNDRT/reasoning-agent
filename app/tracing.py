# app/tracing.py
# Safe tracer that records prompts and sanitized outputs without exposing raw chain-of-thought.

import re

_COT_HINTS = re.compile(
    r"(?i)(chain[-\s]?of[-\s]?thought|think step by step|let'?s think step by step|reason step by step)"
)

def _chunk_to_text(chunk):
    """
    Extract text from many possible streaming chunk shapes:
    - OpenAI (python client): chunk.choices[0].delta.content / message.content
    - OpenAI (raw dict): {"choices":[{"delta":{"content":"..."}}]}
    - Ollama: {"message": {"content": "..."}}
    - Google (genai): candidates[0].content.parts[*].text
    - Fallbacks: .text / .content / str(chunk)
    """
    # 1) direct attribute
    t = getattr(chunk, "text", None)
    if isinstance(t, str) and t:
        return t

    # 2) OpenAI python client objects
    try:
        choices = getattr(chunk, "choices", None)
        if choices and len(choices) > 0:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                dc = getattr(delta, "content", None)
                if isinstance(dc, str) and dc:
                    return dc
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                mc = getattr(msg, "content", None)
                if isinstance(mc, str) and mc:
                    return mc
    except Exception:
        pass

    # 3) dict-like
    if isinstance(chunk, dict):
        # Ollama
        if "message" in chunk and isinstance(chunk["message"], dict):
            c = chunk["message"].get("content")
            if isinstance(c, str) and c:
                return c
        # OpenAI raw dict
        ch = chunk.get("choices")
        if isinstance(ch, list) and ch:
            delta = ch[0].get("delta") if isinstance(ch[0], dict) else None
            if isinstance(delta, dict):
                dc = delta.get("content")
                if isinstance(dc, str) and dc:
                    return dc
            msg = ch[0].get("message") if isinstance(ch[0], dict) else None
            if isinstance(msg, dict):
                mc = msg.get("content")
                if isinstance(mc, str) and mc:
                    return mc
        # Google genai
        cand = chunk.get("candidates")
        if isinstance(cand, list) and cand:
            content = cand[0].get("content") if isinstance(cand[0], dict) else None
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list) and parts:
                    texts = [p.get("text") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
                    if texts:
                        return "".join(texts)
        # generic keys
        tx = chunk.get("text")
        if isinstance(tx, str) and tx:
            return tx
        cx = chunk.get("content")
        if isinstance(cx, str) and cx:
            return cx

    # 4) object with 'content' attribute
    cattr = getattr(chunk, "content", None)
    if isinstance(cattr, str) and cattr:
        return cattr

    # 5) stringifiable fallback
    try:
        s = str(chunk)
        if s and not s.startswith("<") and not s.endswith(">"):
            return s
    except Exception:
        pass
    return ""

def _redact_prompt(prompt: str, max_chars: int = 2000) -> str:
    """Redact explicit CoT triggers and truncate long prompts."""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    redacted = _COT_HINTS.sub("[REDACTED_CoT_HINT]", prompt)
    # Drop lines that look like explicit 'Reasoning:' sections
    redacted = re.sub(r"(?im)^\s*(reasoning|deliberation)\s*:\s*.*$", "[REDACTED_REASONING_LINE]", redacted)
    return redacted[:max_chars]

def _sanitize_output(text: str) -> str:
    """Keep a safe summary: last A/B/C/D if present, else a short snippet."""
    if not isinstance(text, str):
        text = str(text)
    letters = re.findall(r"\b([ABCD])\b", text)
    if letters:
        return letters[-1]
    snippet = re.sub(r"\s+", " ", text).strip()
    return snippet[:200]

class TracingLLM:
    """
    Wraps an LLM object (from models.registry.build_llm) and records prompts & sanitized outputs.
    Compatible with .generate() and .generate_stream().
    """
    def __init__(self, llm):
        self.llm = llm
        self.last = {"prompts": [], "outputs": []}

    # Non-streaming
    def generate(self, prompt: str, **kwargs):
        # (unchanged) – keep your existing redact/sanitize here
        red_prompt = _redact_prompt(prompt)
        self.last["prompts"].append(red_prompt)
        out = self.llm.generate(prompt, **kwargs)
        text = getattr(out, "text", out if isinstance(out, str) else str(out))
        self.last["outputs"].append(_sanitize_output(text))
        return out

    # Streaming
    def generate_stream(self, prompt: str, **kwargs):
        """
        Yield STRING chunks (not raw SDK objects). Robust to StopIteration-return patterns.
        Also collect a safe summary for the dev trace.
        """
        self.last["prompts"].append(_redact_prompt(prompt))
        letters_seen, final_text_parts = [], []

        # Fallback to non-streaming if backend lacks streaming
        if not hasattr(self.llm, "generate_stream"):
            try:
                out = self.llm.generate(prompt, **kwargs)
                text = getattr(out, "text", out if isinstance(out, str) else str(out))
            except Exception:
                text = ""
            if text:
                final_text_parts.append(text)
                letters_seen.extend(re.findall(r"\b([ABCD])\b", text))
                yield text  # yield a single string chunk
        else:
            try:
                iterator = self.llm.generate_stream(prompt, **kwargs)
                for raw in iterator:
                    text = _chunk_to_text(raw)
                    if text:
                        final_text_parts.append(text)
                        letters_seen.extend(re.findall(r"\b([ABCD])\b", text))
                        yield text  # IMPORTANT: yield string chunks
            except (StopIteration, RuntimeError):
                # Some SDKs 'return value' from generators -> surfaces as RuntimeError per PEP 479
                pass

        # Summarize for the dev trace
        if letters_seen:
            self.last["outputs"].append(letters_seen[-1])
        elif final_text_parts:
            self.last["outputs"].append(_sanitize_output("".join(final_text_parts)))
        else:
            self.last["outputs"].append("[no text chunks]")
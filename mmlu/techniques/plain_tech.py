# mmlu/techniques/plain_tech.py
import re

LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def build_prompt(row, letter_only: bool = True) -> str:
    """
    Minimal 'plain' prompt for MMLU:
      - Question text
      - Choices labeled A/B/C/D
      - (Optional) one short line requesting a single-letter answer
    """
    q = (row.get("question") or "").strip()
    choices = row.get("choices") or []
    lines = [q, ""]
    for lbl, text in zip("ABCD", choices):
        lines.append(f"{lbl}) {text}")
    if letter_only:
        lines.append("")
        lines.append("Answer with a single letter: A, B, C, or D.")
    return "\n".join(lines).strip()


LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def _extract_letter(text: str) -> str:
    """
    Extract the first valid multiple-choice letter (A–D).
    Falls back to uppercase string if no clean letter is found.
    """
    if not text:
        return ""
    
    # If model outputs like "D) stamina" or "Answer: C) gravity"
    # → strip everything after the letter
    m = LETTER_RE.search(text)
    if m:
        return m.group(1).upper()

    # Fallback: if text starts with a letter+paren (A) or B) etc.)
    if text.strip()[:2].upper() in ["A)", "B)", "C)", "D)"]:
        return text.strip()[0].upper()

    # Otherwise, just uppercase the raw text (not ideal, but safe)
    return text.strip().upper()

def run(row, llm, temperature: float = 0.0, letter_only: bool = True):
    """
    Non-streaming plain technique:
      - Just ask the question with its choices.
      - No few-shot, no CoT, no self-ask.
      - If the LLM wrapper supports `system=None`, we disable system prompts.
    """
    prompt = build_prompt(row, letter_only=letter_only)
    try:
        out = llm.generate(prompt, temperature=temperature, n=1, system=None)
    except TypeError:
        # Wrapper doesn't support system=None, fallback to normal
        out = llm.generate(prompt, temperature=temperature, n=1)

    text = (out.texts[0] if getattr(out, "texts", None) else str(out or "")).strip()
    yhat = _extract_letter(text)
    return yhat, {"prompt": prompt, "raw": text}


def stream(row, llm, temperature: float = 0.0, letter_only: bool = True):
    """
    Streaming plain technique:
      - Yields partial raw text every ~8 chunks.
      - Final yield includes the extracted answer and raw text.
    """
    prompt = build_prompt(row, letter_only=letter_only)
    yield {"event": "status", "msg": "Streaming raw model output (plain)..."}

    buf = []
    try:
        gen = llm.generate_stream(prompt, temperature=temperature, system=None)
    except TypeError:
        gen = llm.generate_stream(prompt, temperature=temperature)

    for chunk in gen:
        if not chunk:
            continue
        buf.append(chunk)
        if len(buf) % 8 == 0:  # every ~8 chunks, show something
            yield {"event": "partial", "text": "".join(buf[-8:])}

    final_text = "".join(buf)
    answer = _extract_letter(final_text)
    yield {"event": "final", "answer": answer, "raw": final_text}

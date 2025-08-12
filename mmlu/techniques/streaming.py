# mmlu/techniques/streaming.py
import re, time
from collections import Counter
from typing import Iterator, Tuple, Dict, Any
from mmlu.prompts import build_cot, extract_letter

LETTER = re.compile(r"\b([ABCD])\b")

def self_ask_stream(item, llm, max_steps=4, temperature=0.3) -> Iterator[Dict[str, Any]]:
    """
    Yields dicts like:
    {"event":"step", "step":k, "subq":"...", "proposed":"C"}
    and finally: {"event":"final", "answer":"C"}
    """
    notes, last = "", None
    TEMPLATE = """We will solve by asking ourselves brief sub-questions.
Current problem:
{q}

Options:
A) {A}
B) {B}
C) {C}
D) {D}

So far:
{notes}

Now ask ONE helpful sub-question then propose ONE letter.
Format:
SubQ: ...
Proposed: <A|B|C|D>
"""
    for k in range(1, max_steps+1):
        prompt = TEMPLATE.format(q=item["question"], A=item["choices"][0],
                                 B=item["choices"][1], C=item["choices"][2],
                                 D=item["choices"][3], notes=notes or "(none)")
        # We don't need token-by-token here; one shot per step:
        out = "".join(list(llm.generate_stream(prompt, temperature=temperature))) or \
              llm.generate(prompt, temperature=temperature).texts[0]
        subq_m = re.search(r"SubQ:\s*(.*)", out)
        prop_m = re.search(r"Proposed:\s*([ABCD])", out)
        subq = subq_m.group(1).strip() if subq_m else "(no sub-question)"
        letter = prop_m.group(1).upper() if prop_m else "A"
        notes += f"\nSubQ: {subq}\nProposed: {letter}"
        yield {"event":"step","step":k,"subq":subq,"proposed":letter}
        if letter == last:
            yield {"event":"final","answer":letter}
            return
        last = letter
    yield {"event":"final","answer": last or "A"}

def self_consistency_stream(item, llm, n=7, temperature=0.8) -> Iterator[Dict[str, Any]]:
    """
    Runs N independent samples (stream per sample optional) and yields a live vote tally.
    Yields: {"event":"vote","i":i,"letter":"B","tally":{"A":x,"B":y,"C":z,"D":w}}
    Finally: {"event":"final","answer":"B","tally":{...}}
    
    Robust version: if provider streaming fails or yields nothing,
    fall back to a non-streaming generate() for that sample.
    """
    tally = Counter()
    prompt = build_cot(item)

    for i in range(1, n + 1):
        partial = []
        try:
            stream = llm.generate_stream(prompt, temperature=temperature)
            for chunk in stream:
                if isinstance(chunk, str) and chunk:
                    partial.append(chunk)
        except StopIteration:
            # Some SDKs can surface this — ignore and fallback
            pass
        except Exception:
            # Any other streaming error → fallback to non-stream
            partial = []

        if partial:
            out = "".join(partial)
        else:
            # Fallback non-stream call (never raises StopIteration)
            try:
                out = llm.generate(prompt, temperature=temperature).texts[0]
            except Exception:
                out = ""

        letter = extract_letter(out or "")
        tally[letter] += 1
        yield {"event": "vote", "i": i, "letter": letter,
               "tally": {k: tally.get(k, 0) for k in "ABCD"}}

    final = tally.most_common(1)[0][0] if tally else "A"
    yield {"event": "final", "answer": final,
           "tally": {k: tally.get(k, 0) for k in "ABCD"}}

def cot_sanitized_stream(item, llm, temperature=0.0) -> Iterator[Dict[str, Any]]:
    """
    Streams partial tokens but only emits neutral milestones, not raw CoT text.
    Yields: {"event":"status","msg":"analyzing question..."} etc.
    Finally yields: {"event":"final","answer":"D"}
    
    Neutral progress messages; robust to streaming failures.
    """
    yield {"event": "status", "msg": "analyzing question…"}
    prompt = build_cot(item)
    buf = ""

    try:
        for chunk in llm.generate_stream(prompt, temperature=temperature):
            if isinstance(chunk, str):
                buf += chunk
                if len(buf) > 200:
                    yield {"event": "status", "msg": "considering options…"}
                if "Final answer:" in buf:
                    break
    except StopIteration:
        pass
    except Exception:
        buf = ""  # force fallback below

    if "Final answer:" not in buf:
        try:
            buf = llm.generate(prompt, temperature=temperature).texts[0]
        except Exception:
            buf = ""

    letter = extract_letter(buf or "")
    yield {"event": "status", "msg": "selecting best option…"}
    yield {"event": "final", "answer": letter}

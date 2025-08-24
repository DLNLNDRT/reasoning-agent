import numpy as np, time
from datasets import load_dataset
from evaluate import load as load_metric
from models.registry import build_llm
from mmlu.techniques import few_shot, cot, self_consistency, self_ask, plain_tech

ACC = load_metric("accuracy")

def tiny_exemplars(subject, k=3):
    try:
        ds = load_dataset("cais/mmlu", subject)["validation"]
    except:
        return []
    ex = []
    for i in range(min(k, len(ds))):
        row = ds[i]
        ex.append({
            "id": f"{subject}_ex_{i}",
            "q": row["question"],
            "A": row["choices"][0], "B": row["choices"][1],
            "C": row["choices"][2], "D": row["choices"][3],
            "y": row["answer"]
        })
    return ex

def run_subject(subject, technique, n_items=25, **params):
 #   from mmlu.techniques.plain import run as plain_run
    # Build LLM from explicit provider/model
    provider = params.pop("provider")
    model = params.pop("model")
    llm = build_llm(provider, model)

    ds = load_dataset("cais/mmlu", subject)["test"].select(range(n_items))
    preds, refs, lat = [], [], []
    exemplars = tiny_exemplars(subject, k=params.pop("k_shots", 3)) if technique == "few_shot" else None

    for row in ds:
        t0 = time.perf_counter()
        if technique == "plain":
            yhat, _ = plain_tech.run(row, llm, temperature=params.get("temperature", 0.0), letter_only=True)
        elif technique == "few_shot":
            yhat, trace = few_shot.run(row, llm, exemplars=exemplars)
        elif technique == "cot":
            yhat, trace = cot.run(row, llm)
        elif technique == "self_consistency":
            yhat, trace = self_consistency.run(
                row, llm, n=params.get("n", 7), temperature=params.get("temperature", 0.8)
            )
        elif technique == "self_ask":
            yhat, trace = self_ask.run(
                row, llm, max_steps=params.get("max_steps", 4), temperature=params.get("temperature", 0.3)
            )
        else:
            raise ValueError("unknown technique")
        lat.append(time.perf_counter() - t0)
        preds.append(yhat); 
        refs.append(row["answer"])

    acc = ACC.compute(predictions=preds, references=refs)["accuracy"]
    return {"subject": subject, 
            "acc": acc, 
            "p50_latency": float(np.median(lat)), 
            "n": len(refs)
            }

# Map multiple-choice letters to numeric indices for the accuracy metric
_LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}

def _to_idx_list(xs):
    """Convert list of 'A'/'B'/'C'/'D' (or messy strings) to 0/1/2/3 for metrics."""
    out = []
    for x in xs:
        s = str(x).strip().upper()
        out.append(_LETTER_TO_IDX.get(s, -1))  # -1 for invalid predictions
    return out

def run_subject_iter(subject, technique, n_items=25, **params):
    """
    Yields per-item progress while evaluating a subject AND (optionally) logs results.
    Extra params (optional):
      - run_id: str
      - logger: eval.logger.Logger
      - provider: str, model: str, family: str
      - trace_enabled: bool
      - technique hyperparams: n, max_steps, k_shots, temperature
    """
    from datasets import load_dataset
    from evaluate import load as load_metric
    import time, numpy as np, re

    from models.registry import build_llm
    from mmlu.techniques import few_shot, cot, self_consistency, self_ask, plain_tech
    from .run import tiny_exemplars  # reuse

    # ---------- helpers: robust letter normalization ----------
    LETTERS = "ABCD"
    LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

    def _to_letter_any(x):
        """
        Return 'A'/'B'/'C'/'D' from:
          - already a letter (any case),
          - 0..3 index (zero-based),
          - 1..4 index (one-based),
          - strings like '2', '1.0', or 'C)' etc.
        Fallback: uppercased first A-D found; else uppercased string.
        """
        if x is None:
            return ""
        s = str(x).strip().upper()
        if s in {"A", "B", "C", "D"}:
            return s
        # e.g., "C) stamina"
        if len(s) >= 2 and s[0] in LETTERS and s[1] in {")", ".", ":", "-", " "}:
            return s[0]
        # numeric-ish
        try:
            i = int(float(s))
            if i in (0, 1, 2, 3):
                return LETTERS[i]
            if i in (1, 2, 3, 4):
                return LETTERS[i - 1]
        except Exception:
            pass
        # look for any standalone A-D
        m = LETTER_RE.search(s)
        if m:
            return m.group(1).upper()
        return s

    def _letter_to_idx(L):
        """A/B/C/D -> 0/1/2/3; tolerant of odd inputs."""
        L = _to_letter_any(L)
        return {"A": 0, "B": 1, "C": 2, "D": 3}.get(L, -1)

    def _to_idx_list(seq):
        """Convert any list of letters/numbers to 0..3 ints for metric."""
        out = []
        for x in seq:
            idx = _letter_to_idx(x)
            if idx < 0:
                # fallback: try numeric
                try:
                    idx = int(float(str(x).strip()))
                except Exception:
                    idx = -1
            out.append(idx)
        return out
    # ---------------------------------------------------------

    # tracer
    try:
        from app.tracing import TracingLLM
    except Exception:
        TracingLLM = None  # allow running even if import path differs

    run_id = params.pop("run_id", None)
    logger = params.pop("logger", None)
    family = params.get("family")  # may be None
    trace_enabled = params.pop("trace_enabled", False)

    ACC = load_metric("accuracy")

    provider = params.pop("provider")
    model = params.pop("model")
    llm = build_llm(provider, model)

    tracer = None
    if trace_enabled and TracingLLM is not None:
        tracer = TracingLLM(llm)
        llm = tracer

    # dataset slice
    ds = load_dataset("cais/mmlu", subject)["test"].select(range(n_items))
    # pre compute exemplars for few-shot once per subject
    exemplars = tiny_exemplars(subject, k=params.pop("k_shots", 3)) if technique == "few_shot" else None

    preds, refs, times = [], [], []

    # start event
    yield {"event": "start", "subject": subject, "n": len(ds)}

    for idx, row in enumerate(ds, start=1):
        t0 = time.perf_counter()

        # track tracer counts to slice per-item snippets
        prev_p = len(tracer.last["prompts"]) if tracer else 0
        prev_o = len(tracer.last["outputs"]) if tracer else 0

        # -------- run the chosen technique --------
        if technique == "plain":
            yhat, _ = plain_tech.run(
                row, llm,
                temperature=params.get("temperature", 0.0),
                letter_only=True
            )
        elif technique == "few_shot":
            yhat, _ = few_shot.run(row, llm, exemplars=exemplars)
        elif technique == "cot":
            yhat, _ = cot.run(row, llm)
        elif technique == "self_consistency":
            yhat, _ = self_consistency.run(
                row, llm,
                n=params.get("n", 7),
                temperature=params.get("temperature", 0.8)
            )
        elif technique == "self_ask":
            yhat, _ = self_ask.run(
                row, llm,
                max_steps=params.get("max_steps", 4),
                temperature=params.get("temperature", 0.3)
            )
        else:
            # Unknown technique: ask plain question once
            raw = llm.generate(row["question"]).texts[0]
            yhat = _to_letter_any(raw)
        # -----------------------------------------

        elapsed = time.perf_counter() - t0

        # Normalize BOTH pred and ref to LETTERS for UI/logging
        pred_letter = _to_letter_any(yhat)
        ref_letter  = _to_letter_any(row.get("answer"))
        is_correct  = int(pred_letter == ref_letter)

        preds.append(pred_letter)
        refs.append(ref_letter)
        times.append(elapsed)

        # Gather per-item trace snippets
        prompt_snip = output_snip = None
        if tracer:
            new_prompts = tracer.last["prompts"][prev_p:]
            new_outputs = tracer.last["outputs"][prev_o:]
            if new_prompts:
                prompt_snip = new_prompts[-1]
            if technique == "plain":
                output_snip = pred_letter  # ensure final letter
            else:
                output_snip = new_outputs[-1] if new_outputs else pred_letter

        # UI event (per-item)
        yield {
            "event": "item",
            "i": idx,
            "pred": pred_letter,            # letter for UI
            "ref": ref_letter,              # letter for UI
            "elapsed": elapsed,
            "correct": is_correct,
            "prompt_snippet": prompt_snip,  # may be None
            "output_snippet": output_snip,  # may be None
        }

        # Log this item if logger/run_id provided
        if logger and run_id:
            logger.append({
                "timestamp": None,
                "run_id": run_id,
                "mode": "batch",
                "provider": provider,
                "model": model,
                "family": family,
                "technique": technique,
                "subject": subject,
                "question_index": idx - 1,  # zero-based within this slice
                "prediction": pred_letter,  # <-- LETTER
                "reference": ref_letter,    # <-- LETTER
                "correct": is_correct,
                "latency_sec": round(elapsed, 4),
                "self_consistency_n": params.get("n"),
                "self_ask_steps": params.get("max_steps"),
                "few_shot_k": (params.get("k_shots") if technique == "few_shot" else None),
                "prompt_snippet": prompt_snip,
                "output_snippet": output_snip,
            })

    # Convert to indices ONLY for the metric computation
    preds_idx = _to_idx_list(preds)
    refs_idx  = _to_idx_list(refs)
    acc = ACC.compute(predictions=preds_idx, references=refs_idx)["accuracy"]

    summary = {
        "subject": subject,
        "acc": acc,
        "p50_latency": float(np.median(times)) if times else None,
        "n": len(refs),
    }

    yield {"event": "end", "subject": subject, "summary": summary}

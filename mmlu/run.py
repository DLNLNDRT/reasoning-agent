import numpy as np, time
from datasets import load_dataset
from evaluate import load as load_metric
from models.registry import build_llm
from mmlu.techniques import few_shot, cot, self_consistency, self_ask
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
    # Build LLM from explicit provider/model
    provider = params.pop("provider")
    model = params.pop("model")
    llm = build_llm(provider, model)

    ds = load_dataset("cais/mmlu", subject)["test"].select(range(n_items))
    preds, refs, lat = [], [], []
    exemplars = tiny_exemplars(subject, k=params.pop("k_shots", 3)) if technique == "few_shot" else None

    for row in ds:
        t0 = time.perf_counter()
        if technique == "few_shot":
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
        preds.append(yhat); refs.append(row["answer"])

    acc = ACC.compute(predictions=preds, references=refs)["accuracy"]
    return {"subject": subject, "acc": acc, "p50_latency": float(np.median(lat)), "n": len(refs)}

# 
def run_subject_iter(subject, technique, n_items=25, **params):
    """
    Yields per-item progress while evaluating a subject AND (optionally) logs results.
    Extra params (optional):
      - run_id: str
      - logger: eval.logger.Logger
      - provider: str, model: str, family: str, and technique hyperparams
    """
    from datasets import load_dataset
    from evaluate import load as load_metric
    import time, numpy as np

    from models.registry import build_llm
    from mmlu.techniques import few_shot, cot, self_consistency, self_ask
    from .run import tiny_exemplars  # reuse

    run_id = params.pop("run_id", None)
    logger = params.pop("logger", None)
    family = params.get("family")  # may be None

    ACC = load_metric("accuracy")

    provider = params.pop("provider")
    model = params.pop("model")
    llm = build_llm(provider, model)

    ds = load_dataset("cais/mmlu", subject)["test"].select(range(n_items))
    exemplars = tiny_exemplars(subject, k=params.pop("k_shots", 3)) if technique == "few_shot" else None

    preds, refs, times = [], [], []
    yield {"event": "start", "subject": subject, "n": len(ds)}

    for idx, row in enumerate(ds, start=1):
        t0 = time.perf_counter()
        if technique == "few_shot":
            yhat, _ = few_shot.run(row, llm, exemplars=exemplars)
        elif technique == "cot":
            yhat, _ = cot.run(row, llm)
        elif technique == "self_consistency":
            yhat, _ = self_consistency.run(row, llm, n=params.get("n", 7), temperature=params.get("temperature", 0.8))
        elif technique == "self_ask":
            yhat, _ = self_ask.run(row, llm, max_steps=params.get("max_steps", 4), temperature=params.get("temperature", 0.3))
        else:
            raise ValueError("unknown technique")
        elapsed = time.perf_counter() - t0

        preds.append(yhat); refs.append(row["answer"]); times.append(elapsed)
        yield {"event": "item", "i": idx, "pred": yhat, "ref": row["answer"], "elapsed": elapsed}

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
                "question_index": idx - 1,  # zero-based index in this slice
                "prediction": yhat,
                "reference": row["answer"],
                "correct": int(yhat == row["answer"]),
                "latency_sec": round(elapsed, 4),
                "self_consistency_n": params.get("n"),
                "self_ask_steps": params.get("max_steps"),
                "few_shot_k": None,  # exemplars were fixed per-subject; you can store it if desired
            })

    acc = ACC.compute(predictions=preds, references=refs)["accuracy"]
    summary = {"subject": subject, "acc": acc, "p50_latency": float(np.median(times)), "n": len(refs)}
    yield {"event": "end", "subject": subject, "summary": summary}

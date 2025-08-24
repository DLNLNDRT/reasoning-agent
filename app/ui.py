# app/ui.py
# Streamlit UI for the MMLU reasoning agent with live/instant solving and clear in-UI explanations.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.tracing import TracingLLM  # or: from tracing import TracingLLM

import uuid
import pandas as pd
import streamlit as st
from datasets import get_dataset_config_names, load_dataset

from eval.logger import Logger
from models.registry import build_llm
from mmlu.techniques import few_shot, cot, self_consistency, self_ask, plain_tech
from mmlu.run import run_subject, tiny_exemplars, run_subject_iter

TECHNIQUE_COLORS = {
    "plain": "#808080",            # Gray baseline
    "few_shot": "#1f77b4",         # Blue
    "cot": "#ff7f0e",              # Orange
    "self_consistency": "#2ca02c", # Green
    "self_ask": "#d62728",         # Red
}

# --- Robust letter conversion: handles A/B/C/D, 0-3, 1-4, ints/floats/strings/numpy ---
def _to_letter_any(x):
    """
    Return 'A'/'B'/'C'/'D' from either:
      - already a letter (case-insensitive),
      - a 0..3 index (zero-based),
      - a 1..4 index (one-based),
      - numeric-looking strings ('0', '1.0', etc.), numpy scalars.
    Falls back to uppercased string.
    """
    if x is None:
        return ""
    s = str(x).strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    # try numeric paths
    try:
        # accept '0', '1', '2', '3', '4', '1.0', numpy types, etc.
        i = int(float(s))
        if i in (0, 1, 2, 3):      # zero-based
            return "ABCD"[i]
        if i in (1, 2, 3, 4):      # one-based
            return "ABCD"[i - 1]
    except Exception:
        pass
    return s

def _safe_join_lines(items, limit=None):
    """Join a list of items into lines, skipping Nones and empty strings."""
    seq = [str(x) for x in (items[-limit:] if (limit and items) else items or []) if x is not None and str(x) != ""]
    return "\n".join(seq) if seq else "(none)"


st.set_page_config(page_title="Reasoning Agent (MMLU)", layout="wide")

# ---------- Subjects ----------
# Using the actively maintained 'cais/mmlu' dataset.
SUBJECTS = [s for s in get_dataset_config_names("cais/mmlu") if s[:1].isalpha()]

# =================== SIDEBAR ===================
st.sidebar.header("Frontier model")
st.sidebar.caption("Pick the hosted provider/model used when you choose the Frontier family below.")

frontier_provider = st.sidebar.selectbox(
    "Provider",
    ["openai", "google"],
    index=0,
    help="OpenAI = gpt-4o-mini. Google = gemini-1.5-flash."
)

frontier_model = st.sidebar.selectbox(
    "Model",
    (["gpt-4o-mini", "gpt-5"] if frontier_provider == "openai"
     else ["gemini-1.5-flash", "gemini-2.5-pro"]),
    index=(1 if frontier_provider == "openai" else 1),  # default to gpt-5 for OpenAI, 2.5-pro for Google
    help=("OpenAI: gpt-4o-mini (low cost) or gpt-5 (best quality).  "
          "Google: gemini-1.5-flash (fast/cheap) or gemini-2.5-pro (best reasoning).")
)


# --- Title with model name ---
st.title(f"Reasoning Agent — MMLU with Transparent Traces")
st.caption(
    f"Currently using **{frontier_provider}:{frontier_model}** as the frontier model.  \n"
    "Compare frontier (hosted) and small (local) models on the MMLU benchmark using four prompting techniques. "
    "Enable *Live mode* to watch sanitized reasoning updates (votes/steps/status) while the model thinks."
)

st.sidebar.header("Small model (Ollama)")
st.sidebar.caption("Pick the local model (run `ollama serve` and `ollama pull <model>` beforehand).")
small_model = st.sidebar.selectbox(
    "Model",
    ["gemma2:9b", "llama3:8b"],
    index=0, # default is gemma
    help="Local model served by Ollama to represent the 'small' family."
)

st.sidebar.header("Technique & batch")
st.sidebar.caption("Choose the prompting technique and how many questions to use in batch runs.")
technique = st.sidebar.selectbox(
    "Technique",
    ["plain", "few_shot", "cot", "self_consistency", "self_ask"],
    index=0, # default is plain
    help=(
        "plain: ask the question directly with no reasoning techniques "
        "few_shot: prepend a few solved examples.  "
        "cot: hidden step-by-step reasoning with a final letter.  "
        "self_consistency: sample N times and majority-vote.  "
        "self_ask: ask brief sub-questions before deciding."
    )
)
n_items = st.sidebar.slider(
    "Items per subject (batch)",
    10, 100, 25, 5,
    help="Number of questions per subject for Batch Evaluation."
)

# Technique hyperparams
n_sc = st.sidebar.slider(
    "Self-consistency: N samples", 3, 15, 7, 1,
    help="How many diverse samples to generate and vote on."
) if technique == "self_consistency" else None

steps = st.sidebar.slider(
    "Self-ask: max steps", 2, 6, 4, 1,
    help="Maximum number of self-questions before finalizing the answer."
) if technique == "self_ask" else None

kshots = st.sidebar.slider(
    "Few-shot: K exemplars", 0, 5, 3, 1,
    help="How many solved examples to prepend to the prompt."
) if technique == "few_shot" else None

model_family = st.sidebar.radio(
    "Use which family to answer?",
    ["Frontier", "Small"],
    index=1, # default is small
    help="Determines which model family answers in both Single Question and Batch tabs."
)

trace_enabled = st.sidebar.checkbox(
    "Show the prompts and output",
    value=True, # default is true, show the prompt and output
    help="Displays redacted prompts sent to the model and sanitized outputs (A/B/C/D or short snippet)."
)

st.sidebar.header("Help")
st.sidebar.caption(
    "• Set API keys in your shell (OPENAI_API_KEY / GOOGLE_API_KEY).  "
    "• For local models, run `ollama serve` and pull models.  "
    "• Switch techniques and models freely; Live mode shows sanitized progress."
)

# Build live LLM for single-question mode
if model_family == "Frontier":
    llm = build_llm(frontier_provider, frontier_model)
    chosen = f"{frontier_provider}:{frontier_model}"
else:
    llm = build_llm("ollama", small_model)
    chosen = f"ollama:{small_model}"

tracer = None
if trace_enabled:
    llm = TracingLLM(llm)
    tracer = llm

# =================== MAIN TABS ===================
tab1, tab2, tab3 = st.tabs(
    ["Single Question", "Batch Evaluation", "Results & Analysis"]
)
# ------------- Tab 1 -------------
with tab1:
    st.subheader("Single Question")
    st.caption(
    "Default technique = **plain** (ask directly, no prompting tricks). "
    "Compare with few-shot, chain-of-thought (CoT), self-consistency, or self-ask. "
    "Toggle **Live mode** to watch sanitized reasoning breadcrumbs while it thinks."
    )

    subj = st.selectbox(
        "Subject",
        SUBJECTS,
        index=(SUBJECTS.index("astronomy") if "astronomy" in SUBJECTS else 0),
        help="Subject domain within MMLU."
    )

    test = load_dataset("cais/mmlu", subj)["test"]
    i = st.number_input(
        "Question index", 0, len(test) - 1, 0, 1,
        help="Which question from this subject to display."
    )
    row = test[i]

    st.markdown("### Question")
    st.caption("The model must return exactly one letter: A, B, C, or D.")
    st.write(row["question"])
    for idx, opt in zip("ABCD", row["choices"]):
        st.write(f"**{idx})** {opt}")

    # Live toggle + single Solve button
    live_mode = st.toggle(
        "Live mode (stream reasoning)", value=False,
        help="When ON, shows live, sanitized breadcrumbs (votes/steps/status) while the model thinks."
    )
    solve_clicked = st.button(
        "Solve", type="primary",
        help="Run the selected technique with the chosen model family."
    )

    result_box = st.empty()
    trace_box = st.container()

    if solve_clicked:
        if live_mode:
            st.info(f"Using **{chosen}** with **{technique}** (live).")
            timeline = st.expander("Live reasoning timeline", expanded=True)
            live_holder = timeline.empty()
            working = st.status("Working… streaming reasoning.", state="running")

            final_answer, gt = None, None

            if trace_enabled and tracer is not None:
                tracer.reset()

            if technique == "plain":
                # build prompt: question + choices
                q = (row.get("question") or "").strip()
                choices = row.get("choices") or []
                prompt = "\n".join([q, ""] + [f"{lbl}) {text}" for lbl, text in zip("ABCD", choices)] +
                                ["", "Answer with a single letter: A, B, C, or D."]).strip()

                buffer = []
                try:
                    stream = llm.generate_stream(prompt, temperature=0.0, system=None)
                except TypeError:
                    stream = llm.generate_stream(prompt, temperature=0.0)

                for chunk in stream:
                    if chunk:
                        buffer.append(chunk)
                        if len(buffer) % 4 == 0:
                            live_holder.markdown("".join(buffer))

                full_text = "".join(buffer)
                final_answer = plain_tech._extract_letter("".join(buffer))
                gt = _to_letter_any(row.get("answer"))

            elif technique == "few_shot":
                ex = tiny_exemplars(subj, k=kshots if kshots else 3)
                yhat, _ = few_shot.run(row, llm, exemplars=ex)
                final_answer = _to_letter_any(yhat)
                gt = _to_letter_any(row.get("answer"))

            elif technique == "self_ask":
                from mmlu.techniques.streaming import self_ask_stream
                steps_md = []
                for evt in self_ask_stream(row, llm, max_steps=steps if steps else 4):
                    if evt["event"] == "step":
                        steps_md.append(f"**Step {evt['step']}** — {evt['subq']} → {evt['proposed']}")
                        live_holder.markdown("\n\n".join(steps_md))
                    elif evt["event"] == "final":
                        final_answer = _to_letter_any(evt.get("answer"))
                        gt = _to_letter_any(row.get("answer"))

            elif technique == "self_consistency":
                from mmlu.techniques.streaming import self_consistency_stream
                for evt in self_consistency_stream(row, llm, n=n_sc if n_sc else 7, temperature=0.8):
                    if evt["event"] == "vote":
                        live_holder.markdown(f"Votes so far: {evt['tally']}")
                    elif evt["event"] == "final":
                        final_answer = _to_letter_any(evt.get("answer"))
                        gt = _to_letter_any(row.get("answer"))

            else:  # cot
                from mmlu.techniques.streaming import cot_sanitized_stream
                status = []
                for evt in cot_sanitized_stream(row, llm):
                    if evt["event"] == "status":
                        status.append(f"- {evt['msg']}")
                        live_holder.markdown("\n".join(status))
                    elif evt["event"] == "final":
                        final_answer = _to_letter_any(evt.get("answer"))
                        gt = _to_letter_any(row.get("answer"))

            # ✅ Unified result output
            if final_answer is not None:
                mark = "✅" if final_answer == gt else "❌"
                result_box.success(f"Final answer: **{final_answer}** · Correct Answer: **{gt}** {mark}")

                # ✅ Centralized logging
                try:
                    Logger().append({
                        "timestamp": None,
                        "run_id": str(uuid.uuid4()),
                        "mode": "single",
                        "provider": frontier_provider if model_family == "Frontier" else "ollama",
                        "model": frontier_model if model_family == "Frontier" else small_model,
                        "family": "frontier" if model_family == "Frontier" else "small",
                        "technique": technique,
                        "subject": subj,
                        "question_index": int(i),
                        "prediction": final_answer,
                        "reference": gt,
                        "correct": int(final_answer == gt),
                        "latency_sec": None,
                        "self_consistency_n": n_sc if technique == "self_consistency" else None,
                        "self_ask_steps": steps if technique == "self_ask" else None,
                        "few_shot_k": kshots if technique == "few_shot" else None,
                        "prompt_snippet": (tracer.last["prompts"][-1] if (tracer and tracer.last["prompts"]) else None),
                        "output_snippet": final_answer #(tracer.last["outputs"][-1] if (tracer and tracer.last["outputs"]) else None),
                    })
                except Exception:
                    pass

                # ✅ Unified Developer Trace
                if trace_enabled and tracer is not None:
                    with st.expander("Developer Trace (sanitized)", expanded=False):
                        st.markdown("**Prompt(s):**")
                        st.code("\n\n---\n\n".join(tracer.last["prompts"][-3:]) or "(none)")
                        st.markdown("**Outputs (last up to 10):**")
                        st.code("\n".join(tracer.last["outputs"][-10:]) or "(none)")

            working.update(label="Done", state="complete")


        else:
            # ----- INSTANT (non-streaming) path -----
            st.info(f"Using **{chosen}** with **{technique}** (instant).")
            with st.spinner("Solving…"):
                final_answer, gt, trace = None, None, {}

                if trace_enabled and tracer is not None:
                    tracer.reset()

                if technique == "plain":
                    from mmlu.techniques import plain_tech
                    yhat, t = plain_tech.run(row, llm, temperature=0.0, letter_only=True)
                    final_answer = _to_letter_any(yhat)
                    gt = _to_letter_any(row.get("answer"))
                    trace = {
                        "plan": {"technique": "plain"},
                        "deliberation": {"summary": "Asked the question directly (no system prompt, no few-shot)."},
                    }

                elif technique == "few_shot":
                    ex = tiny_exemplars(subj, k=kshots if kshots is not None else 3)
                    yhat, _ = few_shot.run(row, llm, exemplars=ex)
                    final_answer = _to_letter_any(yhat)
                    gt = _to_letter_any(row.get("answer"))
                    trace = {
                        "plan": {"technique": "few_shot", "k": len(ex)},
                        "deliberation": {"exemplar_ids": [e["id"] for e in ex]},
                    }

                elif technique == "cot":
                    yhat, _ = cot.run(row, llm)
                    final_answer = _to_letter_any(yhat)
                    gt = _to_letter_any(row.get("answer"))
                    trace = {
                        "plan": {"technique": "cot"},
                        "deliberation": {"summary": "Reasoned privately, then chose a final letter."},
                    }

                elif technique == "self_consistency":
                    n = n_sc if n_sc is not None else 7
                    yhat, t = self_consistency.run(row, llm, n=n)
                    final_answer = _to_letter_any(yhat)
                    gt = _to_letter_any(row.get("answer"))
                    trace = {
                        "plan": {"technique": "self_consistency", "n": n},
                        "votes": t["votes"],
                    }

                elif technique == "self_ask":
                    s = steps if steps is not None else 4
                    yhat, t = self_ask.run(row, llm, max_steps=s)
                    final_answer = _to_letter_any(yhat)
                    gt = _to_letter_any(row.get("answer"))
                    trace = {
                        "plan": {"technique": "self_ask", "steps": s},
                        "breadcrumbs": t["steps"],
                    }

            # ✅ Unified result display
            if final_answer is not None:
                mark = "✅" if final_answer == gt else "❌"
                result_box.success(f"Final answer: **{final_answer}** · Correct Answer: **{gt}** {mark}")

                # ✅ Unified logging
                try:
                    Logger().append({
                        "timestamp": None,
                        "run_id": str(uuid.uuid4()),
                        "mode": "single",
                        "provider": frontier_provider if model_family == "Frontier" else "ollama",
                        "model": frontier_model if model_family == "Frontier" else small_model,
                        "family": "frontier" if model_family == "Frontier" else "small",
                        "technique": technique,
                        "subject": subj,
                        "question_index": int(i),
                        "prediction": final_answer,
                        "reference": gt,
                        "correct": int(final_answer == gt),
                        "latency_sec": None,
                        "self_consistency_n": n_sc if technique == "self_consistency" else None,
                        "self_ask_steps": steps if technique == "self_ask" else None,
                        "few_shot_k": kshots if technique == "few_shot" else None,
                        "prompt_snippet": (tracer.last["prompts"][-1] if (tracer and tracer.last["prompts"]) else None),
                        "output_snippet": final_answer #(tracer.last["outputs"][-1] if (tracer and tracer.last["outputs"]) else None),
                    })
                except Exception:
                    pass

                # ✅ Unified Developer Trace
                if trace_enabled and tracer is not None:
                    with st.expander("Developer Trace (sanitized)", expanded=False):
                        st.markdown("**Prompt(s):**")
                        st.code(_safe_join_lines(tracer.last["prompts"], limit=3), language="markdown")
                        st.markdown("**Outputs (last up to 10):**")
                        st.code(_safe_join_lines(tracer.last["outputs"], limit=10))

                # ✅ Unified Transparency JSON
                trace_box.write("#### Transparency (sanitized)")
                trace_box.caption(
                    "We show plan, exemplars, votes, and self-ask breadcrumbs, but do not display raw chain-of-thought text."
                )
                trace_box.json(trace)


# ------------- Tab 2 -------------
with tab2:
    st.subheader("Batch Evaluation")
    st.caption(
        "Default technique = **plain** (direct question answering).  \n"
        f"Currently running with **{frontier_provider}:{frontier_model}**.  \n"
        "Evaluate multiple questions per subject with live progress. "
        "Bars show per-subject status; the table updates as each subject finishes."
    )

    picks = st.multiselect(
        "Subjects (batch)",
        SUBJECTS,#[:12],
        default=["astronomy", "abstract_algebra"],
        help="Pick one or more subjects. Results show accuracy and median latency per subject."
    )

    run_batch = st.button(
        "Run Batch",
        help="Execute the selected technique over n_items questions per chosen subject."
    )

    if run_batch and picks:
        provider = frontier_provider if model_family == "Frontier" else "ollama"
        model = frontier_model if model_family == "Frontier" else small_model

        logger = Logger()
        run_id = str(uuid.uuid4())

        table_placeholder = st.empty()
        summary_rows = []

        overall = st.progress(0, text="Starting batch…")
        total_subjects = len(picks)
        completed = 0

        for subj in picks:
            tech_color = TECHNIQUE_COLORS.get(technique, "#666666")

            st.markdown(
                f"<span style='color:{tech_color}; font-weight:bold'>[{technique.upper()}]</span> **{subj}**",
                unsafe_allow_html=True
            )
            pbar = st.progress(0, text=f"{subj}: preparing…")
            log_area = st.empty()

            params = {
                "provider": provider,
                "model": model,
                "family": ("frontier" if model_family == "Frontier" else "small"),
                "run_id": run_id,
                "logger": logger,
                "trace_enabled": trace_enabled
            }

            if technique == "self_consistency":
                params["n"] = n_sc if n_sc else 7
            if technique == "self_ask":
                params["max_steps"] = steps if steps else 4
            if technique == "few_shot":
                params["k_shots"] = kshots if kshots is not None else 3

            lines, n_total, done = [], None, 0

            # ✅ iterate with run_subject_iter (logger handles logging internally)
            for evt in run_subject_iter(subj, technique, n_items=n_items, **params):
                if evt["event"] == "start":
                    n_total = evt["n"]
                    pbar.progress(0, text=f"{subj}: 0/{n_total}")

                elif evt["event"] == "item":
                    done = evt["i"]

                    pred = _to_letter_any(evt.get("pred"))
                    gt   = _to_letter_any(evt.get("ref"))
                    mark = "✅" if pred == gt else "❌"

                    # Row summary (color-coded by technique)
                    line = (
                        f"q {evt['i']:>2}/{n_total}: "
                        f"<span style='color:{tech_color}'>predicted **{pred}**</span> · "
                        f"Correct Answer **{gt}** — {evt['elapsed']:.2f}s {mark}"
                    )

                    # Append snippets if trace enabled
                    if trace_enabled:
                        psnip, osnip = evt.get("prompt_snippet"), evt.get("output_snippet")
                        if psnip or osnip:
                            line += "<br/>"
                            if psnip:
                                line += f"<span style='color:gray'>Prompt:</span> {psnip[:180]}{'…' if len(psnip) > 180 else ''}<br/>"
                            if osnip:
                                line += f"<span style='color:gray'>Output:</span> {osnip[:120]}{'…' if len(osnip) > 120 else ''}"

                    lines.append(line)
                    log_area.markdown("<br/><br/>".join(lines[-6:]), unsafe_allow_html=True)
                    pbar.progress(int(100 * done / n_total), text=f"{subj}: {done}/{n_total}")

                elif evt["event"] == "end":
                    pbar.progress(100, text=f"{subj}: done")
                    summary_rows.append(evt["summary"])
                    table_placeholder.dataframe(
                        pd.DataFrame(summary_rows).sort_values("acc", ascending=False),
                        use_container_width=True
                    )

            completed += 1
            overall.progress(
                int(100 * completed / total_subjects),
                text=f"Batch: {completed}/{total_subjects} subjects complete"
            )

        st.success("Batch complete.")
        st.caption("Accuracy = exact match of the answer letter. p50 latency = median time per question.")

        # 🔎 Batch-level developer trace
        if trace_enabled:
            import pandas as pd
            st.markdown("### Batch Developer Trace (last 20 prompts/outputs)")
            df = pd.read_csv("results/log.csv")
            df = df[df["run_id"] == run_id]
            trace_df = df[["subject","question_index","technique","prediction","reference","prompt_snippet","output_snippet"]]
            trace_df = trace_df.dropna(how="all", subset=["prompt_snippet","output_snippet"]).tail(20)
            st.dataframe(trace_df, use_container_width=True)


# ------------- Tab 3 -------------
with tab3:
    st.subheader("Results & Analysis")
    st.caption(
        f"Last selected frontier model: **{frontier_provider}:{frontier_model}**.  \n"
        "Default technique = **plain** (baseline). Use this as a benchmark to compare "
        "with few-shot, CoT, self-consistency, and self-ask. "
        "Explore saved runs. Data is appended to results/log.csv."
    )

    import os
    import altair as alt

    path = "results/log.csv"
    if not os.path.exists(path):
        st.info("No results yet. Run a single question or batch to start logging.")
    else:
        # --- Clear logs button ---
        col1, col2 = st.columns([1,4])
        if col1.button("🗑️ Clear Logs", help="Delete all results in results/log.csv"):
            try:
                open(path, "w").close()  # empties the file but keeps it
                st.success("Logs cleared successfully.")
            except Exception as e:
                st.error(f"Failed to clear logs: {e}")

        df = pd.read_csv(path)

        # Basic cleaning/casting
        for col in ["timestamp", "correct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "latency_sec" in df.columns:
            df["latency_sec"] = pd.to_numeric(df["latency_sec"], errors="coerce")

        # Filters
        colA, colB, colC, colD = st.columns(4)
        f_family = colA.multiselect(
            "Family", sorted(df["family"].dropna().unique()),
            default=list(sorted(df["family"].dropna().unique()))
        )
        f_provider = colB.multiselect(
            "Provider", sorted(df["provider"].dropna().unique()),
            default=list(sorted(df["provider"].dropna().unique()))
        )
        f_model = colC.multiselect(
            "Model", sorted(df["model"].dropna().unique()),
            default=list(sorted(df["model"].dropna().unique()))
        )
        f_tech = colD.multiselect(
            "Technique", sorted(df["technique"].dropna().unique()),
            default=list(sorted(df["technique"].dropna().unique()))
        )

        f_subjects = st.multiselect(
            "Subjects", sorted(df["subject"].dropna().unique()),
            default=list(sorted(df["subject"].dropna().unique()))
        )

        fdf = df[
            df["family"].isin(f_family) &
            df["provider"].isin(f_provider) &
            df["model"].isin(f_model) &
            df["technique"].isin(f_tech) &
            df["subject"].isin(f_subjects)
        ].copy()

        if fdf.empty:
            st.warning("No rows match the selected filters.")
        else:
            # Aggregates
            agg = (
                fdf
                .assign(correct=fdf["correct"].fillna(0))
                .groupby(["family", "provider", "model", "technique", "subject"], as_index=False)
                .agg(
                    acc=("correct", "mean"),
                    n=("correct", "size"),
                    p50_latency=("latency_sec", lambda s: s.dropna().median() if not s.dropna().empty else None)
                )
                .sort_values(["acc", "n"], ascending=[False, False])
            )

            st.markdown("### Accuracy by Model & Technique")
            st.caption("Higher is better (mean over filtered rows). Click legend to isolate traces.")
            technique_colors = {
                "plain": "#808080",            # Gray baseline
                "few_shot": "#1f77b4",         # Blue
                "cot": "#ff7f0e",              # Orange
                "self_consistency": "#2ca02c", # Green
                "self_ask": "#d62728",         # Red
            }

            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X("acc:Q", title="Accuracy"),
                    y=alt.Y("model:N", sort="-x", title="Model"),
                    color=alt.Color(
                        "technique:N",
                        scale=alt.Scale(domain=list(technique_colors.keys()), range=list(technique_colors.values())),
                        legend=alt.Legend(title="Technique")
                    ),
                    tooltip=["family","provider","model","technique","subject","acc","n","p50_latency"]
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### Latency (median) by Model & Technique")
            st.caption("Lower is better (p50 of latency_sec).")
            lat = agg.dropna(subset=["p50_latency"]).copy()
            if not lat.empty:
                chart2 = (
                    alt.Chart(lat)
                    .mark_bar()
                    .encode(
                        x=alt.X("p50_latency:Q", title="p50 latency (s)"),
                        y=alt.Y("model:N", sort="-x", title="Model"),
                        color=alt.Color(
                            "technique:N",
                            scale=alt.Scale(domain=list(technique_colors.keys()), range=list(technique_colors.values())),
                            legend=alt.Legend(title="Technique")
                        ),
                        tooltip=["family","provider","model","technique","subject","acc","n","p50_latency"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart2, use_container_width=True)
            else:
                st.info("No latency recorded yet (single-question mode doesn’t log latency unless you add timers).")

            st.markdown("### Detailed table (filtered)")
            st.dataframe(agg, use_container_width=True)

        # --- Trace sample browser (sanitized) ---
    st.markdown("### Trace Samples (sanitized)")
    st.caption(
        "Browse recent prompt/output snippets for the filtered subset above. "
        "Prompts are redacted for CoT hints; outputs show final letter or a short snippet."
    )

    if fdf.empty:
        st.info("No rows available under current filters.")
    else:
        # Keep only rows with at least one snippet
        trace_df = fdf.copy()
        # Ensure columns exist (older logs may not have the new columns)
        for col in ["prompt_snippet", "output_snippet", "timestamp"]:
            if col not in trace_df.columns:
                trace_df[col] = None

        # Convert timestamp to readable datetime (if numeric epoch)
        ts_numeric = pd.to_numeric(trace_df["timestamp"], errors="coerce")
        trace_df["ts_readable"] = pd.to_datetime(ts_numeric, unit="s", errors="coerce")

        # Order by newest first
        trace_df = trace_df.sort_values(["ts_readable", "timestamp"], ascending=False)

        # Only rows with at least one snippet present
        trace_df = trace_df[(trace_df["prompt_snippet"].notna()) | (trace_df["output_snippet"].notna())].copy()

        if trace_df.empty:
            st.info("No trace snippets found in the filtered results.")
        else:
            col1, col2, col3 = st.columns([1,1,1])
            max_rows = col1.slider("Rows to show", 5, 200, 50, help="Limit how many recent traces are displayed.")
            truncate_to = col2.slider("Trim long fields to N chars", 80, 1000, 260, 20,
                                    help="For readability, long prompt/output snippets are truncated.")
            allow_download = col3.toggle("Enable CSV download", value=False,
                                        help="Allow downloading the filtered trace rows as CSV.")

            # Columns to display
            cols = [
                "ts_readable", "family", "provider", "model", "technique",
                "subject", "question_index", "prediction", "reference",
                "prompt_snippet", "output_snippet"
            ]
            present_cols = [c for c in cols if c in trace_df.columns]

            # Truncate long text for readability in the grid
            def _trim(s: pd.Series) -> pd.Series:
                return s.fillna("").astype(str).apply(lambda x: (x[:truncate_to] + "…") if len(x) > truncate_to else x)

            view = trace_df[present_cols].head(max_rows).copy()
            if "prompt_snippet" in view.columns:
                view["prompt_snippet"] = _trim(view["prompt_snippet"])
            if "output_snippet" in view.columns:
                view["output_snippet"] = _trim(view["output_snippet"])

            # Add a correctness mark column for readability
            if {"prediction", "reference"}.issubset(view.columns):
                def _mark_row(r):
                    p = str(r.get("prediction", "")).strip().upper()
                    g = str(r.get("reference", "")).strip().upper()
                    return "✅" if p == g else "❌"
                view["correct_mark"] = view.apply(_mark_row, axis=1)
                # Optional: reorder columns to show it earlier
                cols_order = ["ts_readable", "family", "provider", "model", "technique",
                            "subject", "question_index", "prediction", "reference",
                            "correct_mark", "prompt_snippet", "output_snippet"]
                view = view[[c for c in cols_order if c in view.columns]]

            st.dataframe(view, use_container_width=True)

            if allow_download:
                csv_bytes = trace_df[present_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download filtered traces as CSV",
                    data=csv_bytes,
                    file_name="trace_samples_filtered.csv",
                    mime="text/csv"
                )

            st.caption(
                "Tip: use the filters above (Family/Provider/Model/Technique/Subjects) to narrow down trace samples. "
                "Older runs may not include trace columns until logging was upgraded."
            )

            # --- Trace density (how much trace data do we have by model/technique?) ---
            st.markdown("### Trace Density")
            st.caption(
                "Counts of logged trace rows (where a prompt or output snippet exists), grouped by model and technique. "
                "Helps you see where you have enough examples for qualitative review."
            )

            # Reuse the same filtered dataframe: fdf
            if 'fdf' in locals() and not fdf.empty:
                td = fdf.copy()

                # Ensure presence of trace columns for older logs
                for col in ["prompt_snippet", "output_snippet"]:
                    if col not in td.columns:
                        td[col] = None

                # Keep only rows that actually have a trace
                td = td[(td["prompt_snippet"].notna()) | (td["output_snippet"].notna())].copy()

                if td.empty:
                    st.info("No traced rows in the current filter selection.")
                else:
                    # Group counts
                    grp = (
                        td.groupby(["family", "provider", "model", "technique"], as_index=False)
                        .size()
                        .rename(columns={"size": "trace_count"})
                        .sort_values("trace_count", ascending=False)
                    )

                    # Show a small table
                    st.dataframe(grp, use_container_width=True)

                    # Bar chart (Altair)
                    import altair as alt
                    chart_td = (
                        alt.Chart(grp)
                        .mark_bar()
                        .encode(
                            x=alt.X("trace_count:Q", title="Trace rows (count)"),
                            y=alt.Y("model:N", sort="-x", title="Model"),
                            color=alt.Color("technique:N", title="Technique"),
                            tooltip=["family", "provider", "model", "technique", "trace_count"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart_td, use_container_width=True)

                    st.caption(
                        "Tip: Use the filters above to focus on a provider/model/subject, then review trace samples. "
                        "If counts are low, run more batch evaluations with 'Show developer trace' enabled."
                    )
            else:
                st.info("No data loaded yet. Run some single questions or batches first.")


            st.markdown("#### How to read these results")
            st.write(
                "- **Accuracy** = mean(correct) over filtered rows. Compare the same **subject** and **technique** across models for fair head-to-head.\n"
                "- **p50 latency** = median per-question wall time (batch runs only, by default). If missing, run batch to record it.\n"
                "- **n** = number of questions included in the aggregate for that row."
            )

            

# app/ui.py
# Streamlit UI for the MMLU reasoning agent with live/instant solving and clear in-UI explanations.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import uuid
import pandas as pd
import streamlit as st
from datasets import get_dataset_config_names, load_dataset

from eval.logger import Logger
from models.registry import build_llm
from mmlu.techniques import few_shot, cot, self_consistency, self_ask
from mmlu.run import run_subject, tiny_exemplars, run_subject_iter

st.set_page_config(page_title="Reasoning Agent (MMLU)", layout="wide")

st.title("Reasoning Agent — MMLU with Transparent Traces")
st.caption(
    "Compare frontier (hosted) and small (local) models on the MMLU benchmark using four prompting techniques. "
    "Enable *Live mode* to watch sanitized reasoning updates (votes/steps/status) while the model thinks."
)

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
    help="OpenAI = ChatGPT family (e.g., gpt-5/gpt-4o/o3). Google = Gemini models."
)
frontier_model = st.sidebar.selectbox(
    "Model",
    ["gpt-5", "gpt-4o", "o3"] if frontier_provider == "openai" else ["gemini-2.5-pro", "gemini-1.5-pro"],
    index=0,
    help="Exact hosted model to query when Frontier is selected."
)

st.sidebar.header("Small model (Ollama)")
st.sidebar.caption("Pick the local model (run `ollama serve` and `ollama pull <model>` beforehand).")
small_model = st.sidebar.selectbox(
    "Model",
    ["gemma2:9b", "llama3:8b"],
    index=0,
    help="Local model served by Ollama to represent the 'small' family."
)

st.sidebar.header("Technique & batch")
st.sidebar.caption("Choose the prompting technique and how many questions to use in batch runs.")
technique = st.sidebar.selectbox(
    "Technique",
    ["few_shot", "cot", "self_consistency", "self_ask"],
    index=2,
    help=(
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
    index=0,
    help="Determines which model family answers in both Single Question and Batch tabs."
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

# =================== MAIN TABS ===================
tab1, tab2, tab3 = st.tabs(
    ["Single Question (Transparency)", "Batch Evaluation", "Results & Analysis"]
)

# ------------- Tab 1 -------------
with tab1:
    st.subheader("Single Question (Transparency)")
    st.caption(
        "Pick a subject and a question. Click **Solve** to get the model's answer. "
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
            # ----- LIVE (streaming) path -----
            st.info(f"Using **{chosen}** with **{technique}** (live).")
            timeline = st.expander("Live reasoning timeline", expanded=True)
            live_holder = timeline.empty()
            working = st.status("Working… streaming sanitized reasoning.", state="running")

            if technique == "few_shot":
                # Few-shot is static: show exemplars and final result
                ex = tiny_exemplars(subj, k=kshots if kshots is not None else 3)
                live_holder.markdown(f"**Exemplars used:** `{[e['id'] for e in ex]}`")
                yhat, _ = few_shot.run(row, llm, exemplars=ex)
                result_box.success(f"Final answer: **{yhat}**")
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
                        "prediction": yhat,
                        "reference": row["answer"],
                        "correct": int(yhat == row["answer"]),
                        "latency_sec": None,
                        "self_consistency_n": n_sc if technique == "self_consistency" else None,
                        "self_ask_steps": steps if technique == "self_ask" else None,
                        "few_shot_k": kshots if technique == "few_shot" else None,
                    })
                except Exception:
                    pass
                working.update(label="Done", state="complete")

            elif technique == "self_ask":
                from mmlu.techniques.streaming import self_ask_stream
                steps_md = []
                final_answer = None
                for evt in self_ask_stream(row, llm, max_steps=steps if steps else 4):
                    if evt["event"] == "step":
                        steps_md.append(
                            f"**Step {evt['step']}** — SubQ: _{evt['subq']}_ → Proposed: **{evt['proposed']}**"
                        )
                        live_holder.markdown("\n\n".join(steps_md))
                    elif evt["event"] == "final":
                        final_answer = evt["answer"]
                        result_box.success(f"Final answer: **{final_answer}**")
                if final_answer is not None:
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
                            "reference": row["answer"],
                            "correct": int(final_answer == row["answer"]),
                            "latency_sec": None,
                            "self_consistency_n": n_sc if technique == "self_consistency" else None,
                            "self_ask_steps": steps if technique == "self_ask" else None,
                            "few_shot_k": kshots if technique == "few_shot" else None,
                        })
                    except Exception:
                        pass
                working.update(label="Done", state="complete")

            elif technique == "self_consistency":
                from mmlu.techniques.streaming import self_consistency_stream
                votes_md = []
                final_answer = None
                for evt in self_consistency_stream(row, llm, n=n_sc if n_sc else 7, temperature=0.8):
                    if evt["event"] == "vote":
                        t = evt["tally"]
                        votes_md = [
                            f"Votes: A={t.get('A',0)}  B={t.get('B',0)}  C={t.get('C',0)}  D={t.get('D',0)}",
                            f"Latest sample #{evt['i']} → **{evt['letter']}**",
                        ]
                        live_holder.markdown("\n\n".join(votes_md))
                    elif evt["event"] == "final":
                        t = evt["tally"]
                        live_holder.markdown(
                            f"**Final tally:** A={t.get('A',0)}  B={t.get('B',0)}  C={t.get('C',0)}  D={t.get('D',0)}"
                        )
                        final_answer = evt["answer"]
                        result_box.success(f"Final answer: **{final_answer}**")
                if final_answer is not None:
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
                            "reference": row["answer"],
                            "correct": int(final_answer == row["answer"]),
                            "latency_sec": None,
                            "self_consistency_n": n_sc if technique == "self_consistency" else None,
                            "self_ask_steps": steps if technique == "self_ask" else None,
                            "few_shot_k": kshots if technique == "few_shot" else None,
                        })
                    except Exception:
                        pass
                working.update(label="Done", state="complete")

            else:  # technique == "cot" (sanitized streaming)
                from mmlu.techniques.streaming import cot_sanitized_stream
                status = []
                final_answer = None
                for evt in cot_sanitized_stream(row, llm):
                    if evt["event"] == "status":
                        status.append(f"- {evt['msg']}")
                        live_holder.markdown("\n".join(status))
                    elif evt["event"] == "final":
                        final_answer = evt["answer"]
                        result_box.success(f"Final answer: **{final_answer}**")
                if final_answer is not None:
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
                            "reference": row["answer"],
                            "correct": int(final_answer == row["answer"]),
                            "latency_sec": None,
                            "self_consistency_n": n_sc if technique == "self_consistency" else None,
                            "self_ask_steps": steps if technique == "self_ask" else None,
                            "few_shot_k": kshots if technique == "few_shot" else None,
                        })
                    except Exception:
                        pass
                working.update(label="Done", state="complete")

        else:
            # ----- INSTANT (non-streaming) path -----
            st.info(f"Using **{chosen}** with **{technique}** (instant).")
            with st.spinner("Solving…"):
                if technique == "few_shot":
                    ex = tiny_exemplars(subj, k=kshots if kshots is not None else 3)
                    yhat, _ = few_shot.run(row, llm, exemplars=ex)
                    trace = {
                        "plan": {"technique": "few_shot", "k": len(ex)},
                        "deliberation": {"exemplar_ids": [e["id"] for e in ex]},
                    }
                elif technique == "cot":
                    yhat, _ = cot.run(row, llm)
                    trace = {
                        "plan": {"technique": "cot"},
                        "deliberation": {"summary": "Reasoned privately, then chose a final letter."},
                    }
                elif technique == "self_consistency":
                    n = n_sc if n_sc is not None else 7
                    yhat, t = self_consistency.run(row, llm, n=n)
                    trace = {
                        "plan": {"technique": "self_consistency", "n": n},
                        "votes": t["votes"],
                    }
                else:  # self_ask
                    s = steps if steps is not None else 4
                    yhat, t = self_ask.run(row, llm, max_steps=s)
                    trace = {
                        "plan": {"technique": "self_ask", "steps": s},
                        "breadcrumbs": t["steps"],
                    }

            result_box.success(f"Final answer: **{yhat}**")
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
                    "prediction": yhat,
                    "reference": row["answer"],
                    "correct": int(yhat == row["answer"]),
                    "latency_sec": None,
                    "self_consistency_n": n_sc if technique == "self_consistency" else None,
                    "self_ask_steps": steps if technique == "self_ask" else None,
                    "few_shot_k": kshots if technique == "few_shot" else None,
                })
            except Exception:
                pass

            trace_box.write("#### Transparency (sanitized)")
            trace_box.caption(
                "We show plan, exemplars, votes, and self-ask breadcrumbs, but do not display raw chain-of-thought text."
            )
            if technique != "few_shot":  # few_shot path above didn't set t if not needed
                trace_box.json(trace)

# ------------- Tab 2 -------------
with tab2:
    st.subheader("Batch Evaluation")
    st.caption(
        "Evaluate multiple questions per subject with live progress. "
        "Bars show per-subject status; the table updates as each subject finishes."
    )

    picks = st.multiselect(
        "Subjects (batch)",
        SUBJECTS[:12],
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
            st.markdown(f"**{subj}**")
            pbar = st.progress(0, text=f"{subj}: preparing…")
            log_area = st.empty()

            params = {
                "provider": provider,
                "model": model,
                "family": ("frontier" if model_family == "Frontier" else "small"),
                "run_id": run_id,
                "logger": logger
            }
            if technique == "self_consistency":
                params["n"] = n_sc if n_sc else 7
            if technique == "self_ask":
                params["max_steps"] = steps if steps else 4
            if technique == "few_shot":
                params["k_shots"] = kshots if kshots is not None else 3

            lines = []
            n_total = None
            done = 0

            for evt in run_subject_iter(subj, technique, n_items=n_items, **params):
                if evt["event"] == "start":
                    n_total = evt["n"]
                    pbar.progress(0, text=f"{subj}: 0/{n_total}")
                elif evt["event"] == "item":
                    done = evt["i"]
                    lines.append(
                        f"q {evt['i']:>2}/{n_total}: predicted **{evt['pred']}** "
                        f"(gt {evt['ref']}) — {evt['elapsed']:.2f}s"
                    )
                    log_area.markdown("\n\n".join(lines[-6:]))
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
        st.caption("Accuracy is exact match of the answer letter. p50 latency is median time per question.")

    elif run_batch:
        st.warning("Please select at least one subject to run.")

# ------------- Tab 3 -------------
with tab3:
    st.subheader("Results & Analysis")
    st.caption(
        "Explore saved runs. Filter by model/technique/subject and compare accuracy and latency. "
        "Data is appended to results/log.csv."
    )

    import os
    import altair as alt

    path = "results/log.csv"
    if not os.path.exists(path):
        st.info("No results yet. Run a single question or batch to start logging.")
    else:
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
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X("acc:Q", title="Accuracy"),
                    y=alt.Y("model:N", sort="-x", title="Model"),
                    color=alt.Color("technique:N", legend=alt.Legend(title="Technique")),
                    tooltip=["family", "provider", "model", "technique", "subject", "acc", "n", "p50_latency"]
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
                        color=alt.Color("technique:N", legend=alt.Legend(title="Technique")),
                        tooltip=["family", "provider", "model", "technique", "subject", "acc", "n", "p50_latency"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart2, use_container_width=True)
            else:
                st.info("No latency recorded yet (single-question mode doesn’t log latency unless you add timers).")

            st.markdown("### Detailed table (filtered)")
            st.dataframe(agg, use_container_width=True)

            st.markdown("#### How to read these results")
            st.write(
                "- **Accuracy** = mean(correct) over filtered rows. Compare the same **subject** and **technique** across models for fair head-to-head.\n"
                "- **p50 latency** = median per-question wall time (batch runs only, by default). If missing, run batch to record it.\n"
                "- **n** = number of questions included in the aggregate for that row."
            )

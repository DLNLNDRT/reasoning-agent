# Reasoning Agent (MMLU)

An interactive Streamlit UI to run and compare reasoning models on the MMLU benchmark using multiple prompting techniques.  
It supports both **Frontier models** (OpenAI, Google) and **Small models** (local Ollama), with single-question transparency mode and batch evaluation with live progress.

---

## Features

- **Single Question (Transparency Mode)**  
  - Watch sanitized reasoning steps live while the model solves a question.  
  - Choose between few-shot, chain-of-thought (CoT), self-consistency, or self-ask prompting.  

- **Batch Evaluation**  
  - Run multiple MMLU questions per subject.  
  - See live per-subject progress bars and logs.  
  - Summarized accuracy and latency statistics.  

- **Results & Analysis**  
  - Automatic logging of all runs to `results/log.csv`.  
  - Filter results by model, provider, technique, or subject.  
  - Charts for accuracy and latency comparisons.

---

## Requirements

- Python 3.9+  
- [Streamlit](https://streamlit.io/)  
- [datasets](https://huggingface.co/docs/datasets)  
- [pandas](https://pandas.pydata.org/)  
- [altair](https://altair-viz.github.io/)  
- Access to:
  - **Frontier Models** — API keys for OpenAI or Google
  - **Small Models** — [Ollama](https://ollama.com/) installed locally

---

## Installation

1. **Clone the repository** (via SSH)
   ```bash
   git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO

2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Set your API keys (replace with your keys)
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_google_key"

5. For local models, install and run Ollama:
Download Ollama -> https://ollama.com/download
Start the server:
ollama serve
Pull the model you want to use:
ollama pull gemma2:9b


Running the App
Launch the Streamlit UI:
streamlit run app/ui.py

How to Use
Tab 1 — Single Question (Transparency)
Select a subject, question index, model family, and prompting technique.

Optionally enable Live mode to see sanitized reasoning steps.

Click Solve to run the query.

Tab 2 — Batch Evaluation
Select one or more subjects.

Set the number of items per subject.

Click Run Batch to evaluate and view live progress.

Tab 3 — Results & Analysis
Explore saved runs from results/log.csv.

Filter by family, provider, model, technique, or subject.

View accuracy and latency charts.
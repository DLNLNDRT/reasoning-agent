# Reasoning Agent (MMLU)

Compares local small models (Ollama) vs frontier models (OpenAI/Gemini) on MMLU using:
- Few-shot
- Chain-of-thought
- Self-consistency
- Self-ask

Transparent traces (plan, votes, steps) without raw chain-of-thought.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=YOUR_OPENAI_KEY
export GOOGLE_API_KEY=YOUR_GOOGLE_KEY
brew install ollama && ollama serve && ollama pull gemma2:9b && ollama pull llama3:8b
streamlit run app/ui.py


---

# Step 8) Create the model registry (OpenAI, Gemini, Ollama)
**File:** `reasoning-agent/models/registry.py`
```bash
cat > models/registry.py << 'PY'
from dataclasses import dataclass
from typing import List, Optional
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
        self.model = GenerativeModel(self.model_name)
    def generate(self, prompt, temperature=0.0, n=1) -> GenOut:
        texts = []
        for _ in range(n):
            resp = self.model.generate_content(
                prompt, generation_config={"temperature": float(temperature)}
            )
            text = getattr(resp, "text", None)
            if text is None and getattr(resp, "candidates", None):
                text = resp.candidates[0].content.parts[0].text
            texts.append(text or "")
        return GenOut(texts=texts)

# ---------- Small: Ollama (local models) ----------
class OllamaChat(LLM):
    def __init__(self, model=None):
        import ollama
        self.ollama = ollama
        self.model = model or os.getenv("SMALL_MODEL", "gemma2:9b")
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
PY


PY

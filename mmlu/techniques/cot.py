from mmlu.prompts import build_cot, extract_letter
def run(item, llm, temperature=0.0):
    prompt = build_cot(item)
    out = llm.generate(prompt, temperature=temperature, n=1).texts[0]
    letter = extract_letter(out)
    trace = {"technique":"cot"}
    return letter, trace

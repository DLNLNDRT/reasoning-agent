from mmlu.prompts import build_few_shot, extract_letter
def run(item, llm, exemplars, temperature=0.0):
    prompt = build_few_shot(item, exemplars)
    out = llm.generate(prompt, temperature=temperature, n=1).texts[0]
    letter = extract_letter(out)
    trace = {"technique":"few_shot","exemplars":[e["id"] for e in exemplars]}
    return letter, trace

from collections import Counter
from mmlu.prompts import build_cot, extract_letter
def run(item, llm, n=7, temperature=0.8):
    outs = llm.generate(build_cot(item), temperature=temperature, n=n).texts
    letters = [extract_letter(o) for o in outs]
    final = Counter(letters).most_common(1)[0][0]
    trace = {"technique":"self_consistency","votes":{k:letters.count(k) for k in "ABCD"}}
    return final, trace

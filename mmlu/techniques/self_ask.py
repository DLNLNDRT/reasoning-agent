import re
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
def parse(out:str):
    subq = re.search(r"SubQ:\s*(.*)", out)
    prop = re.search(r"Proposed:\s*([ABCD])", out)
    return (subq.group(1).strip() if subq else ""), (prop.group(1).upper() if prop else "A")
def run(item, llm, max_steps=4, temperature=0.3):
    notes, last = "", None
    for _ in range(max_steps):
        prompt = TEMPLATE.format(q=item["question"], A=item["choices"][0], B=item["choices"][1],
                                 C=item["choices"][2], D=item["choices"][3], notes=notes or "(none)")
        out = llm.generate(prompt, temperature=temperature, n=1).texts[0]
        subq, letter = parse(out)
        notes += f"\nSubQ: {subq}\nProposed: {letter}"
        if letter == last: break
        last = letter
    trace = {"technique":"self_ask","steps":[s for s in notes.strip().splitlines() if s]}
    return last or "A", trace

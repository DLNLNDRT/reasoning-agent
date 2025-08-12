import re
LETTER_RE = re.compile(r"\b([ABCD])\b")

BASE = """You are an expert test-taker. Think privately if helpful, then return ONLY the final option letter.

Question:
{q}

Options:
A) {A}
B) {B}
C) {C}
D) {D}

Return strictly one letter: A, B, C, or D.
"""

COT = """You are an expert test-taker. Solve step-by-step privately, then output the letter.

Question:
{q}

Options:
A) {A}
B) {B}
C) {C}
D) {D}

Show brief reasoning. Then on a new line write:
Final answer: <A|B|C|D>
"""

def extract_letter(text: str) -> str:
    m = re.search(r"Final answer:\s*([ABCD])", text, re.I)
    if m: return m.group(1).upper()
    m = LETTER_RE.search(text)
    return m.group(1).upper() if m else "A"

def build_few_shot(item, exemplars):
    parts=[]
    for ex in exemplars:
        parts.append(
f"""Example:
Q: {ex['q']}
A) {ex['A']}
B) {ex['B']}
C) {ex['C']}
D) {ex['D']}
Correct answer: {ex['y']}"""
        )
    parts.append(BASE.format(q=item["question"], A=item["choices"][0], B=item["choices"][1],
                             C=item["choices"][2], D=item["choices"][3]))
    return "\n\n".join(parts)

def build_cot(item):
    return COT.format(q=item["question"], A=item["choices"][0], B=item["choices"][1],
                      C=item["choices"][2], D=item["choices"][3])

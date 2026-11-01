# Prompts used in the paper

## Zero-shot Long COVID labeler (every labeler in Table I / Fig. 3(b); `scripts/llm_label_v2.py`, 4-bit variant `scripts/llm_label_4bit.py`; default = Qwen3.8-27B via `scripts/run_qwen38.sh`)

Single user turn, greedy decoding (`temperature=0`, `max_tokens=256`, thinking disabled for Qwen3),
document truncated to its first 10,000 tokens of the labeler's own tokenizer. Shown with a soft wrap;
the actual string is one line with a single space after "Long COVID." and after "as output.":

```
You are a medical resident, read the pubmed document and tell if it is a case report about Long COVID.
Answer yes or no as output. Is the following context is a a case report about long covid? answer yes or no
context:
"""
{document}
"""
```

* `pred` = 1 if the generated text contains `answer: yes`, else if it contains `answer: no` = 0,
  else the first standalone `yes`/`no` word (models answer with a bare "Yes"/"No" in practice).
* `p_yes` = P(yes-token) / (P(yes-token) + P(no-token)) at the generated position holding the answer
  word (vLLM: from the top-20 log-probabilities; the 4-bit script: from the full softmax over yes/no
  token variants); this is the continuous score fed to Platt scaling / PPI.
* The accepted version's run used the same prompt through Unsloth 4-bit inference and recorded hard
  labels only (`data/llm/qwen14b_4bit_hard_*`).

## Treatment / outcome extraction (Sec. V-G; Qwen3-14B, 4-bit, greedy, `max_new_tokens=512`)

System: `You are a clinical text annotator with medical knowledge. Your task is to extract structured
clinical information from case reports.`

User:
```
You are a clinical information extraction system.
Extract the following from the case report below:
- Patient's **age**
- Patient's **sex**
- Any **comorbidities** (e.g., diabetes, hypertension)
- All **treatments** (e.g., medications, procedures, interventions)
- All **outcomes** (e.g., symptoms, complications, clinical changes)
Respond in **valid JSON** format as shown:
{
  "patient_info": {"age": <int>, "sex": "<string>", "comorbidities": [<list>]},
  "treatments": [<list>],
  "outcomes": [<list>]
}
Case report:
---
{document}
---
```
The first `{...}` block of the response is parsed as JSON; `extractions_repeat.jsonl` is a second run
with sampled decoding (`top_p=0.8, top_k=20`).

## Retrieval prototype (embedding screen) and reranker query

Prototype `q` (Qwen3-Embedding): "Long COVID: an infection-associated chronic condition that occurs after
SARS-CoV-2 infection and persists for at least 3 months as a continuous, relapsing-remitting, or
progressive disease affecting one or more organ systems." Reranker instruction: "Given a description of
Long COVID (...), judge whether the case report describes a Long COVID patient."

Regex cues (union screen): `long covid`, `long[- ]?haul`, `post[- ]?acute sequelae`, `\bpasc\b`,
`post[- ]?covid`, `post[- ]?acute covid`, `chronic covid`, `persistent symptom`, `post[- ]?viral`,
`sequelae of (sars|covid)` (case-insensitive).

"""Zero-shot LLM labeler with a CONTINUOUS yes/no score (ICDM 2026 paper 2, camera-ready).

Uses the paper's ORIGINAL Long COVID prompt (the one that produced the Table I zero-shot
row), greedy decoding, and records for every document:
  pred   : hard label parsed from the generated text ("answer: yes/no")
  p_yes  : normalized P(yes)/(P(yes)+P(no)) read from the token log-probs at the position
           where the model emits its answer word (the continuous score fed to Platt/PPI)
  text   : the generated rationale (short)
Usage:  CUDA_VISIBLE_DEVICES=2 python llm_label_v2.py Qwen/Qwen3-14B qwen14b
        LABEL_POOL=0 -> gold only.
"""
import os, sys, json, math, re, time
import numpy as np
import pandas as pd
from common import load_gold, load_pool, RESULTS

MODEL = sys.argv[1]
TAG = sys.argv[2]
MAX_DOC_TOKENS = int(os.environ.get("MAX_DOC_TOKENS", "10000"))
LABEL_POOL = os.environ.get("LABEL_POOL", "1") == "1"
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.88"))

# ---- the original prompt (Unsloth_qwen14B.ipynb, cell 14) ------------------------------
SYSTEM = ("You are a medical resident, read the pubmed document and tell if it is a "
          "case report about Long COVID. Answer yes or no as output.")


def build_user(text):
    return (SYSTEM + " Is the following context is a a case report about long covid? "
            "answer yes or no \n" + f"context:\n\"\"\"\n{text}\n\"\"\"\n\n")


def truncate(tok, text, max_tokens):
    ids = tok(str(text), add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return str(text)
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)


YES = ("yes", "y")
NO = ("no", "n")


def _norm(t):
    return re.sub(r"[^a-z]", "", str(t).lower())


def p_yes_from_lp(lp_items):
    """lp_items: iterable of (decoded_token, logprob). Returns normalized P(yes)."""
    yes = no = None
    for tok, lp in lp_items:
        t = _norm(tok)
        if t in YES and (yes is None or lp > yes):
            yes = lp
        if t in NO and (no is None or lp > no):
            no = lp
    if yes is None and no is None:
        return None
    if yes is None:
        return 0.02
    if no is None:
        return 0.98
    m = max(yes, no)
    ey, en = math.exp(yes - m), math.exp(no - m)
    return ey / (ey + en)


def parse(gen):
    txt = gen.text
    low = txt.lower()
    if "answer: yes" in low or "answer:yes" in low or "**answer:** yes" in low:
        hard, how = 1, "answer_tag"
    elif "answer: no" in low or "answer:no" in low or "**answer:** no" in low:
        hard, how = 0, "answer_tag"
    else:
        m = re.search(r"\b(yes|no)\b", low)
        hard = 1 if (m and m.group(1) == "yes") else 0
        how = "first_word" if m else "unparsed"
    # continuous score at the answer position
    p_yes, src = None, "none"
    if gen.logprobs:
        toks = gen.token_ids
        dec = []
        for i, lpd in enumerate(gen.logprobs):
            tid = toks[i] if i < len(toks) else None
            dec.append(lpd[tid].decoded_token if (tid is not None and tid in lpd) else "")
        cands = [i for i, d in enumerate(dec) if _norm(d) in YES + NO]
        best = None
        for i in cands:  # prefer a yes/no token that follows "answer"
            prev = _norm("".join(dec[max(0, i - 6):i]))
            if prev.endswith("answer") or "answer" in prev[-12:]:
                best = i
        if best is None and cands:
            best = cands[-1] if how == "answer_tag" else cands[0]
        if best is not None:
            p_yes = p_yes_from_lp((v.decoded_token, v.logprob) for v in gen.logprobs[best].values())
            src = "logprob"
    if p_yes is None:
        p_yes, src = float(hard), "hard_fallback"
    return hard, p_yes, how, src, txt.strip().replace("\n", " ")[:400]


def run(llm, tok, sp, texts, is_qwen3):
    msgs = [[{"role": "user", "content": build_user(truncate(tok, t, MAX_DOC_TOKENS))}] for t in texts]
    kw = dict(add_generation_prompt=True)
    if is_qwen3:
        kw["chat_template_kwargs"] = {"enable_thinking": False}
    outs = llm.chat(msgs, sp, **kw)
    rows = [parse(o.outputs[0]) for o in outs]
    return pd.DataFrame(rows, columns=["pred", "p_yes", "parse", "score_src", "text"])


def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    t0 = time.time()
    g = load_gold(); p = load_pool()
    tok = AutoTokenizer.from_pretrained(MODEL)
    is_qwen3 = "qwen3" in MODEL.lower()
    extra = json.loads(os.environ.get("LLM_EXTRA_JSON", "{}"))   # e.g. {"limit_mm_per_prompt": {"image": 0, "video": 0}} for VL checkpoints
    kw = dict(model=MODEL, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL, max_model_len=12288,
              enforce_eager=os.environ.get("ENFORCE_EAGER", "1") == "1", trust_remote_code=True,
              max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "48")))
    kw.update(extra)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=256, logprobs=20)

    gd = run(llm, tok, sp, list(g.content), is_qwen3)
    gd.insert(0, "label", g.label.values); gd.insert(0, "pmcid", g.pmcid.values)
    gd.to_csv(os.path.join(RESULTS, f"llm_{TAG}_gold.csv"), index=False)
    print(f"[{TAG}] gold done in {time.time()-t0:.0f}s; parse={gd['parse'].value_counts().to_dict()} "
          f"score_src={gd['score_src'].value_counts().to_dict()}", flush=True)

    if LABEL_POOL:
        pdx = run(llm, tok, sp, list(p.content), is_qwen3)
        pdx.insert(0, "pmcid", p.pmcid.values)
        pdx.to_csv(os.path.join(RESULTS, f"llm_{TAG}_pool.csv"), index=False)
        pool_yes = float(pdx.pred.mean()); n_pool = len(pdx)
        print(f"[{TAG}] pool done in {time.time()-t0:.0f}s; parse={pdx['parse'].value_counts().to_dict()} "
              f"score_src={pdx['score_src'].value_counts().to_dict()}", flush=True)
    else:
        pool_yes, n_pool = None, 0

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y = g.label.values; gp = gd.pred.values; gpy = gd.p_yes.values
    summ = dict(model=MODEL, tag=TAG, prompt="original (Unsloth_qwen14B.ipynb cell 14)",
                max_doc_tokens=MAX_DOC_TOKENS, n_gold=len(g), n_pool=n_pool,
                acc=round(accuracy_score(y, gp), 3), macro_f1=round(f1_score(y, gp, average="macro"), 3),
                prec1=round(precision_score(y, gp, zero_division=0), 3),
                rec1=round(recall_score(y, gp, zero_division=0), 3),
                auroc=round(roc_auc_score(y, gpy), 3) if len(set(np.round(gpy, 3))) > 2 else None,
                gold_pred_yes_rate=round(float(gp.mean()), 3),
                pool_pred_yes_rate=None if pool_yes is None else round(pool_yes, 4),
                seconds=round(time.time() - t0))
    with open(os.path.join(RESULTS, f"llm_{TAG}_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()

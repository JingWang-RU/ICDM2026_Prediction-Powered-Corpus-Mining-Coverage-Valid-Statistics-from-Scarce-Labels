"""Replicates the ACCEPTED-version zero-shot labeler exactly — Qwen3-14B loaded in 4-bit NF4
(unsloth default: nf4 + double quant, bf16 compute), the original prompt, greedy decoding,
10,000-token truncation — while additionally recording the continuous yes/no token
probability at the answer position (the score fed to Platt calibration and PPI).
Usage: CUDA_VISIBLE_DEVICES=2 python llm_label_4bit.py [tag]   (default tag qwen14b4bit)
"""
import os, sys, re, json, math, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from common import load_gold, load_pool, RESULTS

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-14B")
TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen14b4bit"
MAX_DOC_TOKENS, MAX_NEW = 10000, 256
LABEL_POOL = os.environ.get("LABEL_POOL", "1") == "1"
SYSTEM = ("You are a medical resident, read the pubmed document and tell if it is a case report about Long COVID. "
          "Answer yes or no as output.")


def build_user(text):
    return SYSTEM + " Is the following context is a a case report about long covid? answer yes or no \n" + f"context:\n\"\"\"\n{text}\n\"\"\"\n\n"


def norm(t): return re.sub(r"[^a-z]", "", str(t).lower())


def main():
    t0 = time.time(); g = load_gold(); p = load_pool()
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0}, torch_dtype=torch.bfloat16).eval()
    print("loaded", MODEL, "in", round(time.time() - t0), "s; mem GB", round(torch.cuda.memory_allocated() / 1e9, 1), flush=True)
    yes_ids = {i for t, i in tok.get_vocab().items() if norm(t) in ("yes", "y")}
    no_ids = {i for t, i in tok.get_vocab().items() if norm(t) in ("no", "n")}

    def label(text):
        ids = tok(str(text), add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_DOC_TOKENS:
            text = tok.decode(ids[:MAX_DOC_TOKENS], skip_special_tokens=True)
        prompt = tok.apply_chat_template([{"role": "user", "content": build_user(text)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 output_scores=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        gen = out.sequences[0, enc["input_ids"].shape[1]:]
        txt = tok.decode(gen, skip_special_tokens=True); low = txt.lower()
        if "answer: yes" in low or "answer:yes" in low: hard, how = 1, "answer_tag"
        elif "answer: no" in low or "answer:no" in low: hard, how = 0, "answer_tag"
        else:
            m = re.search(r"\b(yes|no)\b", low); hard = 1 if (m and m.group(1) == "yes") else 0; how = "first_word" if m else "unparsed"
        # continuous score at the answer token: first generated yes/no token (after "answer" if present)
        toks = [norm(tok.decode([int(t)])) for t in gen]
        cands = [i for i, t in enumerate(toks) if t in ("yes", "y", "no", "n")]
        pos = None
        for i in cands:
            if "answer" in "".join(toks[max(0, i - 6):i]): pos = i
        if pos is None and cands: pos = cands[-1] if how == "answer_tag" else cands[0]
        if pos is not None:
            pr = torch.softmax(out.scores[pos][0].float(), -1)
            py, pn = float(pr[list(yes_ids)].sum()), float(pr[list(no_ids)].sum())
            p_yes = py / (py + pn) if (py + pn) > 0 else float(hard); src = "logprob"
        else:
            p_yes, src = float(hard), "hard_fallback"
        return hard, p_yes, how, src, txt.strip().replace("\n", " ")[:400]

    def run(texts, name):
        rows = []
        for i, t in enumerate(texts):
            rows.append(label(t))
            if i % 100 == 0: print(f"  [{name}] {i}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
        return pd.DataFrame(rows, columns=["pred", "p_yes", "parse", "score_src", "text"])

    gd = run(list(g.content), "gold"); gd.insert(0, "label", g.label.values); gd.insert(0, "pmcid", g.pmcid.values)
    gd.to_csv(os.path.join(RESULTS, f"llm_{TAG}_gold.csv"), index=False)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y = g.label.values
    print(f"[{TAG}] gold: acc {accuracy_score(y, gd.pred):.3f} prec {precision_score(y, gd.pred):.3f} rec {recall_score(y, gd.pred):.3f} "
          f"yes-rate {gd.pred.mean():.3f} parse {gd['parse'].value_counts().to_dict()} ({time.time()-t0:.0f}s)", flush=True)
    pool_yes, n_pool = None, 0
    if LABEL_POOL:
        pdx = run(list(p.content), "pool"); pdx.insert(0, "pmcid", p.pmcid.values)
        pdx.to_csv(os.path.join(RESULTS, f"llm_{TAG}_pool.csv"), index=False); pool_yes, n_pool = float(pdx.pred.mean()), len(pdx)
    summ = dict(model=MODEL + " (4-bit NF4, double-quant, bf16 compute)", tag=TAG, prompt="original (Unsloth_qwen14B.ipynb cell 14)",
                max_doc_tokens=MAX_DOC_TOKENS, n_gold=len(g), n_pool=n_pool, acc=round(accuracy_score(y, gd.pred), 3),
                macro_f1=round(f1_score(y, gd.pred, average="macro"), 3), prec1=round(precision_score(y, gd.pred, zero_division=0), 3),
                rec1=round(recall_score(y, gd.pred, zero_division=0), 3),
                auroc=round(roc_auc_score(y, gd.p_yes), 3) if len(set(np.round(gd.p_yes, 3))) > 2 else None,
                gold_pred_yes_rate=round(float(gd.pred.mean()), 3), pool_pred_yes_rate=None if pool_yes is None else round(pool_yes, 4),
                seconds=round(time.time() - t0))
    json.dump(summ, open(os.path.join(RESULTS, f"llm_{TAG}_summary.json"), "w"), indent=2); print(json.dumps(summ, indent=2), flush=True)


if __name__ == "__main__":
    main()

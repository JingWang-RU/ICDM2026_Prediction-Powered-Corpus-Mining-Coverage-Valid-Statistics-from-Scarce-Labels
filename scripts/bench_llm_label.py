"""Zero-shot LLM labeling of the two truth-known benchmark populations (HoC, ADE).

Purpose: validate PPI coverage with the SAME kind of predictor used in the clinical deployment,
a zero-shot LLM, rather than only the TF-IDF classifier. Emits, for every population sentence,
the hard yes/no decision and the continuous normalized yes/no token probability, in the same
format as llm_label_v2.py.  Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHON=<vllm>=0.17 python> \
        python bench_llm_label.py Qwen/Qwen3.8-27B qwen38_27b hoc ade
"""
import os, sys, json, time, math, re
import numpy as np, pandas as pd
from common import RESULTS

MODEL, TAG = sys.argv[1], sys.argv[2]
WHICH = sys.argv[3:] or ["hoc", "ade"]
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.85"))

TASKS = {
    "hoc": dict(
        system=("You are a careful biomedical annotator. A sentence from a cancer-biology abstract is "
                "labeled YES if it describes a specific hallmark of cancer (sustaining proliferative signaling, "
                "evading growth suppressors, resisting cell death, enabling replicative immortality, inducing "
                "angiogenesis, activating invasion and metastasis, genomic instability and mutation, tumor-promoting "
                "inflammation, cellular energetics, or avoiding immune destruction), and NO if it describes none of them."),
        question="Does this sentence describe a specific hallmark of cancer? answer yes or no"),
    "ade": dict(
        system=("You are a careful biomedical annotator. A sentence from a medical case report is labeled YES if it "
                "reports an adverse drug event, i.e. a harmful or unintended reaction attributed to a drug, and NO otherwise."),
        question="Does this sentence report an adverse drug event? answer yes or no"),
}


def load_pop(which):
    from datasets import load_dataset
    if which == "hoc":
        ds = load_dataset("qanastek/HoC", revision="refs/convert/parquet")
        rows = list(ds["validation"]) + list(ds["test"])
        return [r["text"] for r in rows], np.array([int(any(k != 7 for k in r["label"])) for r in rows])
    ds = load_dataset("SetFit/ade_corpus_v2_classification")
    tr = pd.DataFrame(ds["train"]).drop_duplicates("text"); te = pd.DataFrame(ds["test"]).drop_duplicates("text")
    te = te[~te.text.isin(set(tr.text))]
    return list(te.text), te.label.values.astype(int)


def norm(t): return re.sub(r"[^a-z]", "", str(t).lower())


def p_yes_from_lp(items):
    yes = no = None
    for tok, lp in items:
        t = norm(tok)
        if t in ("yes", "y") and (yes is None or lp > yes): yes = lp
        if t in ("no", "n") and (no is None or lp > no): no = lp
    if yes is None and no is None: return None
    if yes is None: return 0.02
    if no is None: return 0.98
    m = max(yes, no); ey, en = math.exp(yes - m), math.exp(no - m)
    return ey / (ey + en)


def main():
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL, max_model_len=2048,
              enforce_eager=True, trust_remote_code=True, max_num_seqs=int(os.environ.get("MAX_NUM_SEQS", "64")),
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=8, logprobs=20)
    for which in WHICH:
        texts, y = load_pop(which); spec = TASKS[which]
        print(f"[{which}] {len(texts)} sentences, true rate {y.mean():.4f}", flush=True)
        msgs = [[{"role": "user", "content": f"{spec['system']}\n\n{spec['question']}\nsentence:\n\"\"\"\n{t}\n\"\"\"\n\n"}] for t in texts]
        outs = llm.chat(msgs, sp, add_generation_prompt=True, chat_template_kwargs={"enable_thinking": False})
        preds, pys = [], []
        for o in outs:
            gen = o.outputs[0]; low = gen.text.strip().lower()
            m = re.search(r"\b(yes|no)\b", low); hard = 1 if (m and m.group(1) == "yes") else 0
            py = None
            if gen.logprobs:
                toks = gen.token_ids
                dec = [gen.logprobs[i][toks[i]].decoded_token if (i < len(toks) and toks[i] in gen.logprobs[i]) else "" for i in range(len(gen.logprobs))]
                cands = [i for i, d in enumerate(dec) if norm(d) in ("yes", "y", "no", "n")]
                if cands: py = p_yes_from_lp((v.decoded_token, v.logprob) for v in gen.logprobs[cands[0]].values())
            preds.append(hard); pys.append(float(hard) if py is None else py)
        df = pd.DataFrame({"label": y, "pred": preds, "p_yes": pys})
        df.to_csv(os.path.join(RESULTS, f"bench_llm_{which}_{TAG}.csv"), index=False)
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
        summ = dict(model=MODEL, benchmark=which, n=len(y), true_rate=round(float(y.mean()), 4),
                    acc=round(accuracy_score(y, preds), 3), prec1=round(precision_score(y, preds, zero_division=0), 3),
                    rec1=round(recall_score(y, preds, zero_division=0), 3),
                    auroc=round(roc_auc_score(y, pys), 3) if len(set(np.round(pys, 3))) > 2 else None,
                    pred_yes_rate=round(float(np.mean(preds)), 4), naive_bias=round(float(np.mean(preds) - y.mean()), 4),
                    seconds=round(time.time() - t0))
        json.dump(summ, open(os.path.join(RESULTS, f"bench_llm_{which}_{TAG}_summary.json"), "w"), indent=2)
        print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()

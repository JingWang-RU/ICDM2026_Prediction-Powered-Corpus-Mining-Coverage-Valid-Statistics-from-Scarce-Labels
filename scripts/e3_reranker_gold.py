import os, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from common import load_gold, load_embedding_sims, RESULTS

MODEL = "Qwen/Qwen3-Reranker-0.6B"
MAXLEN = 4096
INSTRUCT = ("Given a description of Long COVID (an infection-associated chronic "
            "condition present for at least 3 months after SARS-CoV-2 infection), "
            "judge whether the case report describes a Long COVID patient.")
QUERY = ("Long COVID: an infection-associated chronic condition that occurs after "
         "SARS-CoV-2 infection and persists for at least 3 months as a continuous, "
         "relapsing-remitting, or progressive disease affecting one or more organ systems.")


def main():
    g = load_gold()
    y = g.label.values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
    false_id = tok.convert_tokens_to_ids("no")
    true_id = tok.convert_tokens_to_ids("yes")
    prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements '
              'based on the Query and the Instruct provided. Note that the answer can '
              'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    pre_ids = tok.encode(prefix, add_special_tokens=False)
    suf_ids = tok.encode(suffix, add_special_tokens=False)

    scores = []
    for i, row in g.iterrows():
        doc = str(row.content)[:12000]
        body = f"<Instruct>: {INSTRUCT}\n<Query>: {QUERY}\n<Document>: {doc}"
        ids = tok.encode(body, add_special_tokens=False,
                         truncation=True, max_length=MAXLEN - len(pre_ids) - len(suf_ids))
        ids = pre_ids + ids + suf_ids
        inp = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(inp).logits[0, -1]
        tf = logits[[false_id, true_id]].float()
        prob_yes = torch.softmax(tf, dim=0)[1].item()
        scores.append(prob_yes)
        if i % 25 == 0:
            print(f"  {i}/{len(g)}")
    scores = np.array(scores)

    gold_sim, _ = load_embedding_sims()
    aucs = [average_precision_score(y, gold_sim[k]) for k in range(gold_sim.shape[0])]
    emb = gold_sim[int(np.argmax(aucs))]

    def opmetrics(s):
        p, r, t = precision_recall_curve(y, s)
        mask = r[:-1] >= 0.95
        prec_at_r95 = float(np.max(p[:-1][mask])) if mask.any() else 0.0
        mask2 = p[:-1] >= 0.8
        rec_at_p80 = float(np.max(r[:-1][mask2])) if mask2.any() else 0.0
        return dict(auprc=round(average_precision_score(y, s), 3),
                    auroc=round(roc_auc_score(y, s), 3),
                    prec_at_rec95=round(prec_at_r95, 3),
                    recall_at_prec80=round(rec_at_p80, 3))

    out = dict(model=MODEL, n=len(g),
               embedding=opmetrics(emb), reranker=opmetrics(scores),
               reranker_score_summary=dict(mean=round(float(scores.mean()), 3),
                                           pos_mean=round(float(scores[y == 1].mean()), 3),
                                           neg_mean=round(float(scores[y == 0].mean()), 3)))
    pd.DataFrame({"pmcid": g.pmcid, "label": y, "embed_sim": emb,
                  "reranker": scores}).to_csv(os.path.join(RESULTS, "e3_reranker_gold.csv"), index=False)
    with open(os.path.join(RESULTS, "e3_reranker_gold.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

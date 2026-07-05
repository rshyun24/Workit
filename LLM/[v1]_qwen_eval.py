# -*- coding: utf-8 -*-
import os, re, json
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

BASE     = "Qwen/Qwen3-8B"
ADAPTER  = "/workspace/qwen3-8b-workit-v1/best"
VALID    = "/workspace/data/valid_all.jsonl"
META     = "/workspace/data/valid_all_meta.jsonl"
WRONG_OUT= "/workspace/data/wrong_cases.jsonl"

tok  = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

def parse(txt):
    g = lambda k: (re.search(rf"{k}\s*:\s*(.+)", txt) or [None, None])[1]
    return {k: (g(k).strip() if g(k) else None)
            for k in ["판정", "방향", "유형", "근거", "코멘트"]}

def infer_task(sys_txt, label):
    if label in ("일치", "불일치", "판단보류"):
        return "contract"
    if "결과보고서" in (sys_txt or "") or "이행" in (sys_txt or ""):
        return "report"
    return "deliver"

rows = [json.loads(l) for l in open(VALID, encoding="utf-8")]
meta = None
if os.path.exists(META):
    meta = [json.loads(l) for l in open(META, encoding="utf-8")]
    assert len(meta) == len(rows), "meta/valid 길이 불일치"

recs = []
for i, r in enumerate(rows):
    msgs = r["messages"]
    sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), "")
    gold = parse(msgs[-1]["content"])
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    pred = parse(gen)
    m = meta[i] if meta else {}
    task = m.get("task") or infer_task(sys_txt, gold["판정"])
    tmpl = m.get("template") or m.get("category") or "-"
    recs.append({"i": i, "task": task, "tmpl": tmpl, "gold": gold,
                 "pred": pred, "raw_pred": gen, "user": msgs[-2]["content"]})

def report(subset, title):
    yt = [r["gold"]["판정"] for r in subset]
    yp = [r["pred"]["판정"] or "PARSE_FAIL" for r in subset]
    labels = sorted(set(yt) | set(yp))
    print(f"\n===== {title} (n={len(subset)}) =====")
    print(classification_report(yt, yp, labels=labels, zero_division=0))
    print("confusion (행=정답, 열=예측):", labels)
    print(confusion_matrix(yt, yp, labels=labels))

report(recs, "전체 판정")
for t in sorted(set(r["task"] for r in recs)):
    report([r for r in recs if r["task"] == t], f"task={t}")

NEG = {"불일치", "불가", "미흡"}
cond = [r for r in recs if r["gold"]["판정"] in NEG and r["pred"]["판정"] == r["gold"]["판정"]]
def acc(field):
    if not cond: return None
    return sum(r["pred"][field] == r["gold"][field] for r in cond) / len(cond)
print(f"\n===== 방향·유형 (판정 맞은 불일치류 n={len(cond)}) =====")
print(f"방향 정확도: {acc('방향')}")
print(f"유형 정확도: {acc('유형')}")

def ref_hit(r):
    g, p = r["gold"]["근거"], r["pred"]["근거"]
    if not g or g == "-": return None
    if not p: return False
    toks = [t for t in re.split(r"[,\s]+", g) if t]
    return any(t in p for t in toks)
hits = [h for h in (ref_hit(r) for r in recs) if h is not None]
print(f"\n===== 근거 조항 정확도 (contains, n={len(hits)}) =====")
print(f"근거 적중률: {sum(hits)/len(hits):.3f}" if hits else "대상 없음")

ok = sum(all(r["pred"][k] is not None for k in ["판정", "근거", "코멘트"]) for r in recs)
print(f"\n===== 포맷 준수율 =====")
print(f"필수필드 파싱 성공: {ok}/{len(recs)} = {ok/len(recs):.3f}")

wrong = [r for r in recs if r["pred"]["판정"] != r["gold"]["판정"]]
with open(WRONG_OUT, "w", encoding="utf-8") as f:
    for r in wrong:
        f.write(json.dumps({"task": r["task"], "tmpl": r["tmpl"],
            "gold_판정": r["gold"]["판정"], "pred_판정": r["pred"]["판정"],
            "user": r["user"], "raw_pred": r["raw_pred"]}, ensure_ascii=False) + "\n")
print(f"\n오답 {len(wrong)}건 → {WRONG_OUT}")
print("틀별 오답 분포:")
for k, v in Counter(r["tmpl"] for r in wrong).most_common():
    print(f"  {k}: {v}")

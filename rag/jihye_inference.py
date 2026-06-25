# -*- coding: utf-8 -*-
"""
jihye_inference.py — 계약서 조항 판정 추론 (Qwen3-8B + QLoRA 어댑터)
- 베이스: Qwen/Qwen3-8B  +  학습한 LoRA 어댑터(epoch1 best) = workit_output_v1
- 프롬프트: 학습과 동일 포맷 (train-inference parity)
- 입력: RAG 출력 JSON (clause_text + law_refs[law_name/article_number/chunk_text])
- 출력: 판정/방향/유형/근거/코멘트 (파싱 + 원문)
"""
import os, re, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── 경로 설정 ──
BASE_MODEL_ID = "Qwen/Qwen3-8B"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ↓↓ 다운로드한 best 어댑터 폴더(adapter_config.json 들어있는 곳)로 수정 ↓↓
ADAPTER_PATH = os.path.join(BASE_DIR, "models", "workit_output_v1")

# LOAD_IN_4BIT = True       # GPU 작으면 True(4bit), 여유 있으면 False(bf16)
LOAD_IN_4BIT = False     # 4bit 끄면 bitsandbytes 불필요 → bf16 로드
K_CONTEXT    = 3          # 참고조항 개수 (학습과 동일)
TEXT_MAX     = 300        # 참고조항 본문 컷 (학습과 동일)

# 실제 학습 데이터(train_all.jsonl)의 계약서 시스템 프롬프트와 동일
SYSTEM_PROMPT = ("당신은 지방계약법령에 따라 공공 SW 용역계약서의 조항을 검토하는 전문가입니다. "
                 "검토조항을 참고조항에 비추어 일치/불일치를 판정하되, 참고조항만으로 판단이 불가하면 "
                 "'판단보류'로 답하십시오. 불일치 시 방향(을불리·을유리)·유형(A·B)·근거 조항명·코멘트를 제시하십시오.")


def load_model():
    # 토크나이저는 어댑터 폴더에서 (학습 때 저장된 chat_template 포함 → 프롬프트 일치)
    tok_src = ADAPTER_PATH if os.path.exists(os.path.join(ADAPTER_PATH, "tokenizer_config.json")) else BASE_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    kwargs = dict(device_map="auto", trust_remote_code=True)
    if LOAD_IN_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **kwargs)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def build_user_content(item: dict) -> str:
    """학습(jihye_render.py)과 동일한 user 프롬프트 생성."""
    # RAG 출력엔 seed의 category가 없어 risk_names를 대용 (없으면 기타)
    cat = ", ".join(item.get("risk_names", [])) or "기타"

    refs = "\n".join(
        f"[{i}] {r.get('source_full') or (r.get('law_name','') + ' ' + r.get('article_number','')).strip()}"
        f" — {r.get('chunk_text','')[:TEXT_MAX]}"
        for i, r in enumerate(item.get("law_refs", [])[:K_CONTEXT], 1)
    )
    return (f"카테고리: {cat}\n"
            f"검토조항: {item.get('clause_text','')}\n\n"
            f"참고조항:\n{refs}")


def parse_output(txt: str) -> dict:
    g = lambda k: (re.search(rf"{k}\s*:\s*(.+)", txt) or [None, None])[1]
    return {k: (g(k).strip() if g(k) else None)
            for k in ["판정", "방향", "유형", "근거", "코멘트"]}


def predict(item: dict, model, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(item)},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,        # v4/v5 모두 호환
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
    return {"raw": gen, **parse_output(gen)}


def run_inference(rag_output_path: str, result_path: str = "workit_result.json"):
    with open(rag_output_path, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    print("모델 로드 중...")
    model, tokenizer = load_model()

    final = []
    for item in rag_results:
        if not item.get("law_refs"):
            continue
        cn = item.get("clause_number", "")
        print(f"판정 중: {cn}")
        pred = predict(item, model, tokenizer)
        final.append({
            "clause_number": cn,
            "clause_text": item.get("clause_text", ""),
            "risk_names": item.get("risk_names", []),
            "판정": pred["판정"], "방향": pred["방향"], "유형": pred["유형"],
            "근거": pred["근거"], "코멘트": pred["코멘트"],
            "raw": pred["raw"],
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"완료: {result_path} ({len(final)}건)")
    return final


if __name__ == "__main__":
    run_inference("pdfver_contract_review_output.json")

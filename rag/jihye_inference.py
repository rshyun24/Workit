# import torch
# import json
# import os
# from pathlib import Path
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # 경로 설정
# BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ADAPTER_PATH = os.path.join(BASE_DIR, 'data', 'jihye_sft', 'model_output')

# SYSTEM_PROMPT = "당신은 공공 SW 계약서의 위험 조항을 탐지하는 전문가입니다. 주어진 계약 조항과 참고 기준을 바탕으로 위험 여부를 판단하고 근거를 제시하십시오."

# # 모델 로드_old
# # def load_model():
# #     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
# #     base_model = AutoModelForCausalLM.from_pretrained(
# #         BASE_MODEL_ID,
# #         torch_dtype=torch.float16,
# #         device_map="auto",
# #         trust_remote_code=True,
# #     )
# #     model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
# #     model.eval()
# #     return model, tokenizer

# # 모델 로드_new
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_ID,
#         dtype=torch.float16,
#         device_map="cpu",        # ← auto 대신 cpu로 변경
#         trust_remote_code=True,
#     )
#     model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#     model.eval()
#     return model, tokenizer

# # 단일 조항 판정
# def predict(clause_text: str, law_refs: list, model, tokenizer) -> str:
#     # 참고기준 구성
#     ref_text = "\n".join([
#         f"{r['source_full']}: {r['chunk_text'][:200]}"
#         for r in law_refs[:3]
#     ])

#     user_content = f"다음 계약 조항의 위험 여부를 판단하세요.\n\n[계약조항]\n{clause_text}\n\n[참고기준]\n{ref_text}"

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user_content},
#     ]

#     input_ids = tokenizer.apply_chat_template(
#         messages,
#         return_tensors="pt",
#         add_generation_prompt=True,
#     ).to(model.device)

#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             max_new_tokens=256,
#             do_sample=False,
#         )

#     generated = output[0][input_ids.shape[1]:]
#     return tokenizer.decode(generated, skip_special_tokens=True)

# # RAG 결과 JSON → 판정 결과
# def run_inference(rag_output_path: str, result_path: str = "workit_result.json"):
#     with open(rag_output_path, "r", encoding="utf-8") as f:
#         rag_results = json.load(f)

#     print("모델 로드 중...")
#     model, tokenizer = load_model()

#     final_results = []
#     for item in rag_results:
#         clause_number = item["clause_number"]
#         clause_text = item["clause_text"]
#         law_refs = item["law_refs"]

#         if not law_refs:
#             continue

#         print(f"판정 중: {clause_number}")
#         prediction = predict(clause_text, law_refs, model, tokenizer)

#         final_results.append({
#             "clause_number": clause_number,
#             "clause_text": clause_text,
#             "risk_names": item["risk_names"],
#             "prediction": prediction,
#         })

#     with open(result_path, "w", encoding="utf-8") as f:
#         json.dump(final_results, f, ensure_ascii=False, indent=2)

#     print(f"완료: {result_path}")
#     return final_results


# if __name__ == "__main__":
#     run_inference("contract_review_output.json")


import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 경로 설정
BASE_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_PATH = os.path.join(BASE_DIR, 'data', 'jihye_sft', 'model_output')

SYSTEM_PROMPT = "당신은 공공 SW 계약서의 위험 조항을 탐지하는 전문가입니다. 주어진 계약 조항과 참고 기준을 바탕으로 위험 여부를 판단하고 근거를 제시하십시오."


# transformers==4.49.0(requirements.txt)는 RopeParameters를 모르므로,
# 그 심볼이 추가되기 전(Transformers v5 대응 커밋 이전)의 모델 코드로 고정한다.
BASE_MODEL_REVISION = "8e6fc27d1910b526b5d48a2aa129b08a0293df5e"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=BASE_MODEL_REVISION, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def predict(clause_text: str, law_refs: list, model, tokenizer) -> str:
    ref_text = "\n".join([
        f"{r.get('article') or r.get('source_full', '')}: {r['chunk_text'][:200]}"
        for r in law_refs[:3]
    ])

    user_content = f"다음 계약 조항의 위험 여부를 판단하세요.\n\n[계약조항]\n{clause_text}\n\n[참고기준]\n{ref_text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # tokenize=False로 텍스트 먼저 만들고 따로 토크나이즈
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to('cpu')

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            do_sample=False,
        )

    generated = output[0][input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run_inference(rag_output_path: str, result_path: str = "workit_result.json"):
    with open(rag_output_path, "r", encoding="utf-8") as f:
        rag_results = json.load(f)

    print("모델 로드 중...")
    model, tokenizer = load_model()

    final_results = []
    for item in rag_results:
        clause_number = item["clause_number"]
        clause_text = item["clause_text"]
        law_refs = item["law_refs"]

        if not law_refs:
            continue

        print(f"판정 중: {clause_number}")
        prediction = predict(clause_text, law_refs, model, tokenizer)

        final_results.append({
            "clause_number": clause_number,
            "clause_text": clause_text,
            "risk_names": item["risk_names"],
            "prediction": prediction,
        })

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print(f"완료: {result_path}")
    return final_results


if __name__ == "__main__":
    run_inference("contract_review_output.json")
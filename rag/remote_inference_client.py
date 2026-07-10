# -*- coding: utf-8 -*-
"""
remote_inference_client.py — RunPod 추론 서버(FastAPI) 호출용 로컬 프록시.

CPU 로컬 환경에서 BGE-M3 / bge-reranker-v2-m3 / kanana-1.5-8b를 직접 로드하는 대신,
RunPod에 띄운 추론 서버(main.py, /embed·/rerank·/predict)에 HTTP로 위임한다.

law_rag_pipeline.search_jo()는 model.encode()/reranker.compute_score() 인터페이스만
알면 되므로, 아래 두 클래스는 그 인터페이스를 그대로 흉내 낸다 — search_jo() 쪽
코드는 한 줄도 바꿀 필요가 없다.
"""
import numpy as np
import requests


class RemoteEmbedModel:
    """BGEM3FlagModel.encode()와 같은 인터페이스로 /embed를 호출한다."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def encode(self, texts, return_dense: bool = True, return_sparse: bool = True):
        resp = requests.post(
            f"{self.base_url}/embed",
            json={"texts": list(texts)},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "dense_vecs": np.array(data["dense_vecs"]),
            "lexical_weights": data["lexical_weights"],
        }


class RemoteReranker:
    """FlagReranker.compute_score()와 같은 인터페이스로 /rerank를 호출한다."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def compute_score(self, pairs, normalize: bool = True):
        if not pairs:
            return []
        query = pairs[0][0]
        texts = [p[1] for p in pairs]
        resp = requests.post(
            f"{self.base_url}/rerank",
            json={"query": query, "texts": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["scores"]


def remote_predict(item: dict, base_url: str, timeout: int = 120) -> dict:
    """jihye_inference.predict(item, model, tokenizer)와 같은 반환 형식으로 /predict를 호출한다.

    임베딩/리랭커 서버와 LLM 서버는 transformers 버전이 서로 호환되지 않아
    RunPod에서 별도 venv·별도 포트로 띄운다(embed_server:8000, llm_server:8002).
    그래서 base_url이 RemoteEmbedModel/RemoteReranker와 다를 수 있다.
    """
    resp = requests.post(
        f"{base_url.rstrip('/')}/predict",
        json={"item": item},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["prediction"]

"""
═══════════════════════════════════════════════════════════════════
Workit RAG Pipeline — 토큰 통계 로거
파일명: yoonha_token_stats_logger.py
위치:   Workit/rag/yoonha_token_stats_logger.py
═══════════════════════════════════════════════════════════════════

■ 이 파일의 역할
──────────────────────────────────────────────────────────────────
청킹 전 원본 텍스트의 토큰 수 분포를 측정하고
결과를 Workit/logs/token_stats.json 에 누적 저장합니다.

임베딩(BGE-M3) 모델의 tokenizer를 재사용하므로
별도 tokenizer 로드 비용이 없습니다.

저장되는 통계 항목:
  - timestamp   : 실행 시각
  - source      : 어떤 파일/doc_type 기준인지
  - count       : 섹션(또는 조문) 수
  - mean        : 평균 토큰 수
  - median      : 중앙값
  - p75         : 상위 25% 기준값 (청크 크기 설정 참고용)
  - p95         : 상위 5% 기준값 (MAX_TOKENS 설정 참고용)
  - max         : 최대 토큰 수

■ 연동 대상
──────────────────────────────────────────────────────────────────
  - yoonha_deliver_chunking.py  →  섹션(ParsedSection) 기준 측정
  - law_chunking.py             →  조문(article) 기준 측정

■ 출력 파일
──────────────────────────────────────────────────────────────────
  Workit/logs/token_stats.json
  → 실행할 때마다 append되어 누적됨
  → 파일/폴더 없으면 자동 생성
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/token_stats.json")


# ──────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────
def _load_existing_logs() -> list[dict]:
    """
    기존 로그 파일을 읽어 반환합니다.
    파일이 없거나 비어있으면 빈 리스트를 반환합니다.
    """
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_logs(logs: list[dict]) -> None:
    """
    로그 리스트를 JSON 파일에 저장합니다.
    logs/ 폴더가 없으면 자동 생성합니다.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────
# 핵심 함수
# ──────────────────────────────────────────
def compute_token_stats(texts: list[str], tokenizer) -> dict:
    """
    텍스트 리스트의 토큰 수 분포를 계산합니다.

    임베딩 모델(BGE-M3)의 tokenizer를 그대로 받아 사용하므로
    실제 임베딩 시 적용되는 토큰 기준과 동일합니다.

    Args:
        texts     : 토큰 수를 측정할 텍스트 리스트
                    (deliver → section.text 목록 / law → article text 목록)
        tokenizer : SentenceTransformer 모델의 내장 tokenizer
                    (model.tokenizer 로 전달)

    Returns:
        dict: count, mean, median, p75, p95, max 포함한 통계 딕셔너리
              텍스트가 없으면 빈 딕셔너리 반환
    """
    counts = [len(tokenizer.encode(t)) for t in texts if t.strip()]
    if not counts:
        return {}

    return {
        "count":  len(counts),
        "mean":   round(float(np.mean(counts)), 1),
        "median": round(float(np.median(counts)), 1),
        "p75":    round(float(np.percentile(counts, 75)), 1),
        "p95":    round(float(np.percentile(counts, 95)), 1),
        "max":    int(np.max(counts)),
    }


def log_token_stats(source: str, stats: dict) -> None:
    """
    토큰 통계를 콘솔에 출력하고 logs/token_stats.json 에 누적 저장합니다.

    매 실행마다 기존 로그에 append되므로
    청킹 파이프라인을 반복 실행해도 이전 기록이 유지됩니다.

    Args:
        source : 통계 출처 식별자
                 예) "law:소프트웨어 진흥법" / "deliver:테스트결과보고서"
        stats  : compute_token_stats() 의 반환값
    """
    if not stats:
        print(f"[token_stats] {source}: 측정할 텍스트 없음")
        return

    # 콘솔 출력
    print(
        f"[token_stats] {source} | "
        f"섹션={stats['count']}개 | "
        f"평균={stats['mean']} | 중앙값={stats['median']} | "
        f"p75={stats['p75']} | p95={stats['p95']} | 최대={stats['max']}"
    )

    # JSON 누적 저장
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source":    source,
        **stats,
    }
    logs = _load_existing_logs()
    logs.append(entry)
    _save_logs(logs)
    print(f"[token_stats] 📝 로그 저장 완료 → {LOG_PATH}")


def log_token_stats_from_texts(source: str, texts: list[str], tokenizer) -> dict:
    """
    텍스트 리스트를 받아 통계 계산 + 출력 + 저장을 한 번에 처리합니다.

    compute_token_stats() 와 log_token_stats() 를 묶은 편의 함수입니다.
    chunking 파일에서는 이 함수 하나만 호출하면 됩니다.

    Args:
        source    : 통계 출처 식별자
        texts     : 토큰 수를 측정할 텍스트 리스트
        tokenizer : SentenceTransformer 모델의 내장 tokenizer

    Returns:
        dict: 계산된 통계 (비어있을 수 있음)

    사용 예시:
        # law_chunking.py
        from yoonha_token_stats_logger import log_token_stats_from_texts
        texts = [a["text"] for a in articles if a.get("text")]
        log_token_stats_from_texts(f"law:{law_meta['law_name']}", texts, tokenizer)

        # yoonha_deliver_chunking.py
        from yoonha_token_stats_logger import log_token_stats_from_texts
        texts = [s.text for s in sections if s.text.strip()]
        log_token_stats_from_texts(f"deliver:{doc_type}", texts, tokenizer)
    """
    stats = compute_token_stats(texts, tokenizer)
    log_token_stats(source, stats)
    return stats
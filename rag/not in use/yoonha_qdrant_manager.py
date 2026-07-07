"""
═══════════════════════════════════════════════════════════════════
Workit RAG Pipeline — Qdrant Docker 컨테이너 자동 관리 모듈
파일명: yoonha_qdrant_manager.py
위치:   Workit/rag/yoonha_qdrant_manager.py
═══════════════════════════════════════════════════════════════════

■ 이 파일의 역할
──────────────────────────────────────────────────────────────────
Workit RAG 파이프라인은 벡터 DB로 Qdrant를 사용합니다.
Qdrant는 Docker 컨테이너(workit_qdrant)로 실행되는데,
청킹 스크립트를 돌리기 전에 컨테이너가 켜져 있어야 합니다.

이 파일은 그 과정을 자동화합니다:
  - 컨테이너가 이미 실행 중이면 → 그냥 통과
  - 컨테이너가 꺼져 있으면     → 자동으로 docker start
  - 컨테이너가 아예 없으면     → 생성 방법 안내 후 종료

즉, 매번 Docker Desktop을 열고 컨테이너를 수동으로 켜는 번거로움
없이 청킹 스크립트만 실행하면 Qdrant까지 자동으로 준비됩니다.

■ 연동 대상 파일
──────────────────────────────────────────────────────────────────
  - yoonha_deliver_chunking.py  (산출물 양식 KB 구축)
  - law_chunking.py             (법령 KB 구축)

  두 파일 상단에 아래 두 줄을 추가하면 연동 완료:

      from yoonha_qdrant_manager import ensure_qdrant_running
      ensure_qdrant_running()

■ Docker 컨테이너 정보
──────────────────────────────────────────────────────────────────
  컨테이너명 : workit_qdrant
  이미지     : qdrant/qdrant
  포트       : 6333 (REST API), 6334 (gRPC)
  볼륨 마운트: C:/project/Workit/vectorstore/qdrant_storage
               → 컨테이너 안 /qdrant/storage 에 연결됨
               → 컨테이너를 지워도 데이터는 로컬에 영구 보존

  컨테이너가 없는 경우 아래 명령어로 최초 1회 생성:
      docker run -d --name workit_qdrant \\
        -v C:/project/Workit/vectorstore/qdrant_storage:/qdrant/storage \\
        -p 6333:6333 qdrant/qdrant

■ Qdrant 대시보드 (브라우저에서 확인 가능)
──────────────────────────────────────────────────────────────────
  http://localhost:6333/dashboard

■ 주요 함수
──────────────────────────────────────────────────────────────────
  ensure_qdrant_running()   진입점. 컨테이너 상태 확인 → 필요 시 자동 시작
  _get_container_status()   docker inspect 로 컨테이너 상태 조회
  _start_container()        docker start 로 컨테이너 기동
  _wait_for_ready()         REST API 응답 대기 (최대 15초)
"""

import subprocess
import time
import sys

CONTAINER_NAME = "workit_qdrant"
QDRANT_HOST    = "localhost"
QDRANT_PORT    = 6333


def _get_container_status() -> str | None:
    """
    컨테이너 상태를 문자열로 반환합니다.

    docker inspect 명령어로 컨테이너의 현재 상태를 조회합니다.
    반환값:
        "running"  → 현재 실행 중
        "exited"   → 정상 종료된 상태 (docker stop 등으로 중지됨)
        "created"  → 생성만 되고 한 번도 실행 안 된 상태
        None       → workit_qdrant 컨테이너 자체가 존재하지 않음
    """
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _start_container() -> bool:
    """
    중지된 컨테이너를 시작합니다.

    docker start 명령어를 실행합니다.
    이미 볼륨(-v)과 포트(-p)가 최초 생성 시 설정되어 있으므로
    별도 옵션 없이 start만 해도 동일한 설정으로 재시작됩니다.

    반환값:
        True  → 시작 명령 성공
        False → 시작 실패 (Docker Desktop 미실행 등)
    """
    print(f"[qdrant_manager] '{CONTAINER_NAME}' 시작 중...")
    result = subprocess.run(
        ["docker", "start", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[qdrant_manager] ❌ 시작 실패: {result.stderr.strip()}")
        return False
    return True


def _wait_for_ready(timeout: int = 15) -> bool:
    """
    Qdrant REST API가 응답할 때까지 대기합니다.

    docker start 직후에는 컨테이너 내부 서버가 완전히 뜨기까지
    수 초가 걸립니다. 이 함수는 /collections 엔드포인트에 반복 요청을
    보내며 응답이 올 때까지 기다립니다.

    Args:
        timeout: 최대 대기 시간 (초). 기본값 15초.

    반환값:
        True  → timeout 내에 응답 확인
        False → timeout 초과 (Qdrant 미응답)
    """
    import urllib.request

    url      = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def ensure_qdrant_running() -> None:
    """
    Qdrant 컨테이너 상태를 확인하고 필요 시 자동으로 시작합니다.

    청킹 스크립트(law_chunking.py, yoonha_deliver_chunking.py) 실행 시
    가장 먼저 호출되는 진입 함수입니다.

    동작 흐름:
        1. docker inspect 로 컨테이너 상태 확인
        2. running  → 이미 켜져 있으므로 바로 통과
        3. exited / created → docker start 로 자동 기동 후 API 응답 대기
        4. None (컨테이너 없음) → 생성 명령어 안내 후 sys.exit(1)

    실행 불가 상황에서는 프로그램을 종료하여
    Qdrant 없이 청킹이 진행되는 상황을 방지합니다.
    """
    status = _get_container_status()

    if status == "running":
        print(f"[qdrant_manager] ✅ '{CONTAINER_NAME}' 이미 실행 중")
        return

    if status in ("exited", "created"):
        success = _start_container()
        if not success:
            print(f"[qdrant_manager] ❌ '{CONTAINER_NAME}' 시작 실패")
            print("  → Docker Desktop이 실행 중인지 확인해주세요.")
            sys.exit(1)

        print(f"[qdrant_manager] ⏳ Qdrant 준비 대기 중...")
        if _wait_for_ready():
            print(f"[qdrant_manager] ✅ '{CONTAINER_NAME}' 시작 완료")
        else:
            print(f"[qdrant_manager] ❌ Qdrant 응답 없음 (timeout)")
            print(f"  → http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard 에서 확인해주세요.")
            sys.exit(1)
        return

    if status is None:
        print(f"[qdrant_manager] ❌ '{CONTAINER_NAME}' 컨테이너가 존재하지 않습니다.")
        print("  → 아래 명령어로 컨테이너를 최초 1회 생성해주세요:")
        print(
            f"\n  docker run -d --name {CONTAINER_NAME} "
            f"-v C:/project/Workit/vectorstore/qdrant_storage:/qdrant/storage "
            f"-p {QDRANT_PORT}:6333 qdrant/qdrant\n"
        )
        sys.exit(1)
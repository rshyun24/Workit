# setup_env.py
# 팀원 환경 세팅 스크립트 - 프로젝트 루트에서 python setup_env.py 실행

import os
import subprocess
import sys

# 한국어 Windows 콘솔 기본 코드페이지(cp949)는 ✅/❌ 같은 이모지를 못 담아서
# 리다이렉션(> log.txt) 등으로 콘솔 코드페이지 자동감지가 안 되면 print()에서 죽는다.
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def patch_flagembedding():
    """FlagEmbedding dtype 인자 패치 (transformers 4.49.0 호환)

    FlagEmbedding 1.4.0이 AutoModel.from_pretrained()에 dtype= 키워드를 직접 넘기는데,
    transformers==4.49.0은 이 키워드를 모델 생성자가 그대로 받아 TypeError가 발생한다.
    torch_dtype=으로 바꿔주는 패치.
    """
    try:
        import FlagEmbedding
        flagembedding_dir = os.path.dirname(FlagEmbedding.__file__)
    except ImportError:
        print('⚠️  FlagEmbedding 미설치 - pip install FlagEmbedding 먼저 실행하세요')
        return

    runner_path = os.path.join(
        flagembedding_dir, 'finetune', 'embedder', 'encoder_only', 'm3', 'runner.py'
    )

    if not os.path.exists(runner_path):
        print('⚠️  runner.py 없음 - FlagEmbedding 버전이 다를 수 있습니다. 수동 확인 필요')
        return

    with open(runner_path, 'r', encoding='utf-8') as f:
        content = f.read()

    old = '''        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        )'''
    new = '''        model = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_folder,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )'''

    if old in content:
        content = content.replace(old, new, 1)
        with open(runner_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print('✅ FlagEmbedding runner.py 패치 완료')
    elif new in content:
        print('✅ FlagEmbedding runner.py 이미 패치됨')
    else:
        print('⚠️  FlagEmbedding runner.py 코드가 예상과 다름 - 수동 확인 필요 (버전 차이 가능)')


def check_libreoffice():
    """LibreOffice + H2Orestart (HWP 변환용) 확인"""
    soffice_candidates = [
        r'C:\Program Files\LibreOffice\program\soffice.exe',
        r'C:\Program Files (x86)\LibreOffice\program\soffice.exe',
    ]
    found = next((p for p in soffice_candidates if os.path.exists(p)), None)

    if found:
        print(f'✅ LibreOffice 설치됨 ({found})')
        print('⚠️  H2Orestart 확장 설치 여부는 자동 확인 불가 — HWP 계약서 분석 시 필요')
        print('   미설치 시: https://github.com/ebandal/H2Orestart 에서 .oxt 받아 LibreOffice 확장관리자에 설치')
        print('   (64비트 Java(JRE) 설치 + LibreOffice 옵션에서 Java 런타임 경로 등록 필요)')
    else:
        print('❌ LibreOffice 미설치 - https://www.libreoffice.org/download/download/ 에서 설치')
        print('   (HWP 계약서를 다루지 않는다면 당장은 건너뛰어도 됩니다)')


def check_qdrant_docker():
    """Qdrant Docker 컨테이너 실행 여부 확인"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=qdrant', '--format', '{{.Names}}\t{{.Status}}'],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout.strip()
        if output:
            print(f'✅ Qdrant 컨테이너 실행 중: {output}')
        else:
            print('❌ Qdrant 컨테이너가 떠 있지 않음')
            print('   기존 컨테이너 있으면: docker start <컨테이너이름>')
            print('   없으면 새로 생성:')
            print('   docker run -d --name workit_qdrant -p 6333:6333 -p 6334:6334 \\')
            print('     -v <프로젝트경로>/vectorstore/qdrant_storage:/qdrant/storage qdrant/qdrant')
    except FileNotFoundError:
        print('❌ Docker 미설치 또는 PATH 미등록 - Docker Desktop 설치 필요')
    except Exception as e:
        print(f'⚠️  Qdrant 컨테이너 확인 실패: {e}')


def check_law_kb_export():
    """law_kb 구축용 원본 데이터 파일 확인 (rag/law_upsert_qdrant.py의 DATA_DIR/DATASETS와 동일 경로).

    data/export/{chunks,vectors,sparse_weights}.json은 jo/ho 분리 이전의 옛 포맷
    파일이라 지금은 안 쓰인다 — 실제로 law_upsert_qdrant.py가 읽는 건 data/merged/
    안의 *_jo_fixedid / *_ho_fixedid 파일들이라 그걸 확인한다.
    """
    merged_dir = os.path.join(os.path.dirname(__file__), 'data', 'merged')
    required = [
        'chunks_jo_fixedid.json', 'vectors_jo_fixedid.npz', 'sparse_weights_jo_fixedid.json',
        'chunks_ho_fixedid.json', 'vectors_ho_fixedid.npz', 'sparse_weights_ho_fixedid.json',
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(merged_dir, f))]

    if not missing:
        print('✅ data/merged/ 법령 KB 원본 파일(jo/ho) 모두 존재')
        print('   (Qdrant law_kb_jo/law_kb_ho 컬렉션이 비어있으면 python rag/law_upsert_qdrant.py 실행)')
    else:
        print(f'❌ data/merged/ 에 다음 파일 없음: {", ".join(missing)}')
        print('   구글 드라이브 "청킹 임베딩" 폴더에서 받아서 data/merged/ 에 넣기')


def check_redis():
    """Redis 서버 확인"""
    redis_path = r'C:\Program Files\Redis\redis-cli.exe'
    if os.path.exists(redis_path):
        result = subprocess.run([redis_path, 'ping'], capture_output=True, text=True)
        if 'PONG' in result.stdout:
            print('✅ Redis 실행 중')
        else:
            print('⚠️  Redis 설치됨, 서버 미실행 - redis-server.exe 실행 필요')
    else:
        print('❌ Redis 미설치 - https://github.com/tporadowski/redis/releases 에서 Redis-x64-5.0.14.1.msi 설치')


def check_poppler():
    """poppler 확인"""
    poppler_path = r'C:\poppler-24.08.0\Library\bin\pdftoppm.exe'
    if os.path.exists(poppler_path):
        print('✅ poppler 설치됨')
    else:
        print('❌ poppler 미설치 - https://github.com/oschwartz10612/poppler-windows/releases/tag/v24.08.0-0 에서')
        print('   Release-24.08.0-0.zip 받아서 C:\\poppler-24.08.0\\ 에 압축 풀기')


def check_model():
    """LoRA 어댑터 파일 확인 (rag/jihye_inference.py의 ADAPTER_PATH와 동일 경로).

    RunPod 원격 추론 모드를 쓰면 어댑터는 RunPod 쪽에만 있으면 되고 로컬엔 없어도 된다
    (EMBED_SERVER_URL/LLM_SERVER_URL 설정 시 [0]에서 안내).
    """
    if os.environ.get('EMBED_SERVER_URL') and os.environ.get('LLM_SERVER_URL'):
        print('✅ 원격 추론 모드라 로컬 모델 파일은 필요 없음 (RunPod에 있음)')
        return

    model_path = os.path.join(os.path.dirname(__file__), 'models', 'workit_output', 'adapter_config.json')
    if os.path.exists(model_path):
        print('✅ LoRA 어댑터 파일 존재 (models/workit_output/)')
    else:
        print('❌ LoRA 어댑터 없음 - 구글 드라이브에서 workit_output 폴더를 models/workit_output/ 에 넣기')


def check_remote_inference():
    """RunPod 원격 추론 서버 연동 확인.

    BGE-M3 임베더/리랭커 + kanana LLM을 로컬 CPU에서 직접 로드하는 대신, RunPod GPU에
    이미 띄워둔 추론 서버(embed_server:8000, llm_server:8002)에 HTTP로 위임하는 옵션이다.
    EMBED_SERVER_URL/LLM_SERVER_URL 둘 다 설정하면 자동으로 이 모드로 전환된다
    (contracts/tasks.py의 USE_REMOTE_INFERENCE). CPU보다 훨씬 빠르고, 이 모드를 쓰면
    아래 [1][2]의 로컬 모델 패치도 필요 없다.
    """
    embed_url = os.environ.get('EMBED_SERVER_URL', '').strip()
    llm_url = os.environ.get('LLM_SERVER_URL', '').strip()

    if embed_url and llm_url:
        print(f'✅ 원격 추론 모드 사용 중 (EMBED_SERVER_URL={embed_url}, LLM_SERVER_URL={llm_url})')
        print('   이 모드에서는 아래 [1][2] 로컬 모델 패치가 필요 없습니다 (RunPod에서 이미 로드됨)')
    else:
        print('❌ EMBED_SERVER_URL / LLM_SERVER_URL 미설정 - 로컬 CPU로 직접 추론합니다 (문서 1건에 수십 분)')
        print('   RunPod GPU로 대신 돌리려면 (팀 공용 pod 접속):')
        print('   1. 본인 SSH 공개키를 RunPod 계정에 등록 (없으면: ssh-keygen -t ed25519)')
        print('   2. pod SSH 접속 정보(호스트/포트) 팀장에게 요청 후, 로컬에서 포트포워딩 터널을')
        print('      계속 띄워둘 터미널 하나 확보:')
        print('      ssh -f -N -L 18000:localhost:8000 -L 18002:localhost:8002 \\')
        print('        -p <포트> -i ~/.ssh/id_ed25519 root@<IP>')
        print('   3. celery worker를 켜는 터미널에서 worker 실행 전에:')
        print('      export EMBED_SERVER_URL="http://localhost:18000"')
        print('      export LLM_SERVER_URL="http://localhost:18002"')
        print('   ※ LLM 서버는 요청을 한 번에 하나씩만 처리합니다(동시 요청 시 GPU 상태가 꼬여서')
        print('     직렬화해둠) — 팀원 여러 명이 동시에 분석을 돌리면 뒷사람은 대기하게 됩니다.')
        print('   자세한 셋업 배경/트러블슈팅은 Notion "Workit" 페이지 참고')


def check_qdrant():
    """Qdrant law_kb_jo 컬렉션에 실제 데이터가 들어있는지 확인.

    Qdrant 자체는 로컬 폴더가 아니라 Docker 컨테이너로 띄운다(컨테이너 실행 여부는
    [8] check_qdrant_docker에서 별도 확인) — 컨테이너가 떠 있어도 컬렉션이 비어있을
    수 있어서 API로 직접 포인트 개수를 확인한다.
    """
    import json as _json
    import urllib.request
    import urllib.error

    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = os.environ.get('QDRANT_PORT', '6333')
    url = f'http://{host}:{port}/collections/law_kb_jo'
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = _json.loads(resp.read())
        count = data.get('result', {}).get('points_count', 0)
        if count > 0:
            print(f'✅ Qdrant law_kb_jo 컬렉션에 {count}개 포인트 존재')
        else:
            print('❌ Qdrant law_kb_jo 컬렉션이 비어있음 - python rag/law_upsert_qdrant.py 로 적재 필요')
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        print(f'❌ Qdrant({url}) 연결 실패 - [7]에서 컨테이너가 떠 있는지 먼저 확인하세요')
    except Exception as e:
        print(f'⚠️  Qdrant 컬렉션 확인 중 오류: {e}')


if __name__ == '__main__':
    print('=' * 50)
    print('Workit 환경 세팅 스크립트')
    print('=' * 50)

    print('\n[0] 원격 추론(RunPod) 사용 여부 확인')
    check_remote_inference()

    print('\n[1] FlagEmbedding(BGE-M3) 패치')
    patch_flagembedding()

    print('\n[2] Redis 확인')
    check_redis()

    print('\n[3] poppler 확인')
    check_poppler()

    print('\n[4] LibreOffice (HWP 변환) 확인')
    check_libreoffice()

    print('\n[5] 모델 파일 확인')
    check_model()

    print('\n[6] Qdrant 벡터스토어 확인')
    check_qdrant()

    print('\n[7] Qdrant Docker 컨테이너 확인')
    check_qdrant_docker()

    print('\n[8] 법령 KB 원본 데이터 확인')
    check_law_kb_export()

    print('\n' + '=' * 50)
    print('❌ 항목이 있으면 해당 안내에 따라 설치 후 다시 실행하세요')
    print('✅ 모두 완료되면 아래 순서로 서버 실행:')
    print('  1. docker start <qdrant 컨테이너이름> (또는 docker run으로 신규 생성)')
    print('  2. redis-server.exe 실행')
    print('  3. celery -A config worker --loglevel=info --pool=solo')
    print('  4. python manage.py runserver')
    print('=' * 50)
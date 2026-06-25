# setup_env.py
# 팀원 환경 세팅 스크립트 - 프로젝트 루트에서 python setup_env.py 실행

import os
import subprocess
import sys

def patch_exaone_cache():
    """EXAONE 캐시 파일 패치 (transformers 4.49.0 호환)"""
    base = os.path.expanduser(
        r'~/.cache/huggingface/modules/transformers_modules/'
        r'LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct/'
        r'ccce25bd39c141fe053e0bc75818a8f5fe962802'
    )

    # configuration_exaone.py 패치
    config_path = os.path.join(base, 'configuration_exaone.py')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'from transformers.modeling_rope_utils import RopeParameters' in content and 'try:' not in content:
            content = content.replace(
                'from transformers.modeling_rope_utils import RopeParameters',
                'try:\n    from transformers.modeling_rope_utils import RopeParameters\nexcept ImportError:\n    class RopeParameters: pass'
            )
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print('✅ configuration_exaone.py 패치 완료')
        else:
            print('✅ configuration_exaone.py 이미 패치됨')
    else:
        print('⚠️  configuration_exaone.py 없음 - 모델 첫 실행 시 자동 다운로드 후 다시 실행하세요')

    # modeling_exaone.py 패치
    model_path = os.path.join(base, 'modeling_exaone.py')
    if os.path.exists(model_path):
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
        target = 'from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func'
        if target in content and 'try:' not in content.split(target)[0].split('\n')[-2]:
            content = content.replace(
                target,
                'try:\n    from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func\nexcept ImportError:\n    def use_kernel_forward_from_hub(*a, **kw): return lambda f: f\n    def use_kernel_func_from_hub(*a, **kw): return lambda f: f\n    def use_kernelized_func(*a, **kw): return lambda f: f'
            )
            with open(model_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print('✅ modeling_exaone.py 패치 완료')
        else:
            print('✅ modeling_exaone.py 이미 패치됨')
    else:
        print('⚠️  modeling_exaone.py 없음 - 모델 첫 실행 시 자동 다운로드 후 다시 실행하세요')


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
    """law_kb 구축용 원본 데이터 파일 확인 (yoonha_law_upsert.py 재료)"""
    export_dir = os.path.join(os.path.dirname(__file__), 'data', 'export')
    required = ['chunks.json', 'vectors.npz', 'sparse_weights.json']
    missing = [f for f in required if not os.path.exists(os.path.join(export_dir, f))]

    if not missing:
        print('✅ data/export/ 법령 KB 원본 파일 모두 존재')
        print('   (Qdrant law_kb 컬렉션이 비어있으면 python rag/yoonha_law_upsert.py 실행)')
    else:
        print(f'❌ data/export/ 에 다음 파일 없음: {", ".join(missing)}')
        print('   구글 드라이브에서 받아서 data/export/ 에 넣기')


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
    """모델 파일 확인"""
    model_path = os.path.join(os.path.dirname(__file__), 'data', 'jihye_sft', 'model_output', 'adapter_config.json')
    if os.path.exists(model_path):
        print('✅ 모델 파일 존재')
    else:
        print('❌ 모델 파일 없음 - 구글 드라이브에서 model_output 폴더를 data/jihye_sft/model_output/ 에 넣기')


def check_qdrant():
    """Qdrant 벡터스토어 확인"""
    qdrant_path = os.path.join(os.path.dirname(__file__), 'vectorstore', 'qdrant_storage', 'collection')
    if os.path.exists(qdrant_path):
        print('✅ Qdrant 벡터스토어 존재')
    else:
        print('❌ Qdrant 없음 - 구글 드라이브에서 qdrant_storage 폴더를 vectorstore/ 에 넣기')


if __name__ == '__main__':
    print('=' * 50)
    print('Workit 환경 세팅 스크립트')
    print('=' * 50)

    print('\n[1] EXAONE 캐시 패치')
    patch_exaone_cache()

    print('\n[2] FlagEmbedding(BGE-M3) 패치')
    patch_flagembedding()

    print('\n[3] Redis 확인')
    check_redis()

    print('\n[4] poppler 확인')
    check_poppler()

    print('\n[5] LibreOffice (HWP 변환) 확인')
    check_libreoffice()

    print('\n[6] 모델 파일 확인')
    check_model()

    print('\n[7] Qdrant 벡터스토어 확인')
    check_qdrant()

    print('\n[8] Qdrant Docker 컨테이너 확인')
    check_qdrant_docker()

    print('\n[9] 법령 KB 원본 데이터 확인')
    check_law_kb_export()

    print('\n' + '=' * 50)
    print('❌ 항목이 있으면 해당 안내에 따라 설치 후 다시 실행하세요')
    print('✅ 모두 완료되면 아래 순서로 서버 실행:')
    print('  1. docker start <qdrant 컨테이너이름> (또는 docker run으로 신규 생성)')
    print('  2. redis-server.exe 실행')
    print('  3. celery -A config worker --loglevel=info --pool=solo')
    print('  4. python manage.py runserver')
    print('=' * 50)
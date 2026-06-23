import subprocess
import os
import shutil

# Windows에서 soffice가 PATH에 없는 경우를 위한 폴백 경로들
SOFFICE_CANDIDATES = [
    "soffice",  # PATH에 등록된 경우
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def _find_soffice() -> str:
    for candidate in SOFFICE_CANDIDATES:
        if candidate == "soffice":
            if shutil.which("soffice"):
                return "soffice"
        elif os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "soffice 실행 파일을 찾을 수 없습니다. LibreOffice 설치 경로를 "
        "hwp_converter.py의 SOFFICE_CANDIDATES에 추가하세요."
    )


def convert_hwp_to_pdf(hwp_path: str, output_dir: str) -> str:
    """LibreOffice headless로 HWP를 PDF로 변환"""
    soffice_path = _find_soffice()

    result = subprocess.run([
        soffice_path, "--headless", "--convert-to", "pdf",
        "--outdir", output_dir, hwp_path
    ], capture_output=True, timeout=60)

    pdf_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(hwp_path))[0] + ".pdf"
    )
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"HWP 변환 실패: {result.stderr}")
    return pdf_path
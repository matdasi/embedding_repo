from pathlib import Path

# --- 기본 경로 설정 ---
INPUT_DIR = Path("your pdf path")
OUTPUT_BASE_DIR = Path('/your output path')
TXT_OUTPUT_DIR = OUTPUT_BASE_DIR / "output_text_blocks"
CHROMA_DB_PATH = Path("your chroma db path")

# --- 모델 및 처리 설정 ---
EMBEDDING_MODEL_NAME = "upskyy/bge-m3-korean"
CHUNK_SIZE = 400  # 모델 한계(512)보다 작은 안전한 값
OVERLAP = 40
TABLE_MIN_ROWS = 3
PDF_BATCH_SIZE = 50  # 한 번에 처리할 PDF 파일 수
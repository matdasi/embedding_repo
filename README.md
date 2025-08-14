# PDF 임베딩 파이프라인

PDF 파일에서 텍스트를 추출, 정제하여 ChromaDB에 벡터 임베딩으로 저장하는 스크립트임.

## 1. 설치

프로젝트 실행에 필요한 라이브러리들을 아래 명령어로 설치

```bash
pip install -r requirements.txt
```

## 2. 사용법

1.  **경로 설정**: `config.py` 파일에서 PDF 입력, 결과물 출력, DB 저장 경로를 수정

    ````python
    # filepath: /home/matdasi/private/code/embedding_repo/config.py
    from pathlib import Path

    # --- 기본 경로 설정 ---
    INPUT_DIR = Path("사용자의 PDF 폴더 경로")
    OUTPUT_BASE_DIR = Path('사용자의 결과물 폴더 경로')
    CHROMA_DB_PATH = Path("사용자의 ChromaDB 저장 경로")
    # ...기존 코드...
    ````

2.  **PDF 준비**: `INPUT_DIR`로 지정한 폴더에 PDF 파일을 준비

3.  **실행**: main.py 실행
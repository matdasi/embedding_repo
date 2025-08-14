import os
import pdfplumber
import tabula
import fitz  # PyMuPDF
import traceback

import config
import text_utils

def _process_with_pymupdf(pdf_path: str):
    """PyMuPDF를 사용하여 페이지별 텍스트와 테이블을 추출합니다."""
    pages_content = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        # PyMuPDF의 경미한 오류 무시 옵션 (가능한 경우만)
        try:
            fitz.TOOLS.mupdf_ignore_errors(True)
        except Exception:
            pass

        # 여기서 예외 발생 시 바로 fallback 유도
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"⚠ PyMuPDF로 '{base_name}' 열기 실패 ({e}), pdfplumber로 fallback")
        return None  # 기존 로직 유지: process_pdf_to_pages에서 pdfplumber로 넘어감

    for page_idx, page in enumerate(doc, start=1):
        page_texts = []
        # 1. 테이블 추출 및 마크다운 변환
        tabs = page.find_tables()
        for tab in tabs:
            df = tab.to_pandas()
            if not df.empty and df.shape[0] >= config.TABLE_MIN_ROWS:
                page_texts.append(df.to_markdown(index=False))

        # 2. 텍스트 추출
        raw_text = page.get_text() or ""
        page_texts.append(raw_text)
        
        full_content = "\n".join(page_texts)

        # 3. 페이지 유용성 및 손상 여부 확인
        if not text_utils.is_useless_page(full_content):
            if text_utils.is_text_corrupted(full_content):
                print(f"경고: '{base_name}' (페이지 {page_idx}) 처리 중 PyMuPDF 글자 깨짐 감지. pdfplumber로 재시도합니다.")
                return None  # 기존 로직 유지

            pages_content.append({
                "source": base_name,
                "page": page_idx,
                "content": full_content
            })
            
    return pages_content


def _process_with_pdfplumber(pdf_path: str):
    """pdfplumber를 사용하여 페이지별 텍스트와 테이블을 추출합니다. (Fallback용)"""
    pages_content = []
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            page_texts = []
            # 1. 테이블 추출 (tabula 사용)
            try:
                tables = tabula.read_pdf(pdf_path, pages=str(page_idx), lattice=True, silent=True)
                for df_tbl in tables:
                    if not df_tbl.empty and df_tbl.shape[0] >= config.TABLE_MIN_ROWS:
                        page_texts.append(df_tbl.to_markdown(index=False))
            except Exception:
                pass

            # 2. 텍스트 추출
            raw_text = page.extract_text() or ""
            if not text_utils.is_useless_page(raw_text):
                page_texts.append(raw_text)

            if page_texts:
                pages_content.append({
                    "source": base_name,
                    "page": page_idx,
                    "content": "\n".join(page_texts)
                })
    return pages_content

def process_pdf_to_pages(pdf_path: str):
    """
    단일 PDF 파일을 처리하여 페이지별 콘텐츠를 반환합니다.
    PyMuPDF를 먼저 시도하고, 글자 깨짐 발생 시 pdfplumber로 fallback합니다.
    """
    try:
        # 1. PyMuPDF로 우선 시도
        pymupdf_result = _process_with_pymupdf(pdf_path)
        
        # PyMuPDF가 성공적으로 처리했고, 글자 깨짐이 없었으면 결과를 반환
        if pymupdf_result is not None:
            return (pdf_path, pymupdf_result)
            
        # 2. PyMuPDF 실패 또는 글자 깨짐 감지 시, pdfplumber로 재시도
        pdfplumber_result = _process_with_pdfplumber(pdf_path)
        return (pdf_path, pdfplumber_result)

    except Exception as e:
        print(f"!!!!!! '{pdf_path}' 처리 중 심각한 오류 발생 !!!!!!")
        traceback.print_exc()
        return (pdf_path, [])
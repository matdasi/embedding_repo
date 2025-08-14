import os
import multiprocessing
import traceback
import uuid
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
import torch
import math
import re

import config
import text_utils
from pdf_processor import process_pdf_to_pages


def chunk_text_from_page(page_data, tokenizer):
    """페이지 데이터를 받아 텍스트를 정제하고 문장 단위로 청크를 나눕니다."""
    text = page_data['content']
    
    # 텍스트 정제
    text = text_utils.clean_linebreaks(text)
    text = text_utils.fix_numbering_patterns(text)
    text = text_utils.fix_broken_numbers(text)
    text = text_utils.remove_dot_leaders(text)
    text = text_utils.clean_misparsed_headings(text)
    text = text_utils.remove_page_numbers(text)

    # 문장 단위로 텍스트를 분할하는 정규 표현식
    # 마침표, 물음표, 느낌표 뒤에 공백이나 줄바꿈이 오는 경우를 기준으로 나눕니다.
    sentences = re.split(r'(?<=[.?!])\s+|\n', text)
    
    # 청크의 토큰 길이를 제한하고 오버랩을 적용하여 청크를 생성합니다.
    # 모델의 최대 길이(512)에 가깝게 CHUNK_SIZE를 500으로 설정
    CHUNK_SIZE = 500 
    OVERLAP = 30 # 중복되는 토큰의 수
    
    chunks = []
    current_chunk_tokens = []
    current_chunk_length = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
        
        sentence_tokens = tokenizer.encode(
            sentence,
            add_special_tokens=False,
            truncation=False
        )

        # 현재 청크에 문장을 추가했을 때 CHUNK_SIZE를 초과하는지 확인
        if current_chunk_length + len(sentence_tokens) > CHUNK_SIZE:
            # 현재 청크가 비어 있지 않으면 저장
            if current_chunk_tokens:
                chunks.append(tokenizer.decode(current_chunk_tokens))
                
                # 오버랩 적용을 위해 새로운 청크에 이전 문장들을 추가
                overlap_start = max(0, len(current_chunk_tokens) - OVERLAP)
                current_chunk_tokens = current_chunk_tokens[overlap_start:]
                current_chunk_length = len(current_chunk_tokens)

            # 새로운 청크에 현재 문장 추가
            current_chunk_tokens.extend(sentence_tokens)
            current_chunk_length += len(sentence_tokens)

            # 만약 문장 자체가 CHUNK_SIZE를 초과하면, 이 문장만 별도로 청크로 저장
            if current_chunk_length > CHUNK_SIZE:
                chunks.append(tokenizer.decode(current_chunk_tokens))
                current_chunk_tokens = []
                current_chunk_length = 0
        else:
            current_chunk_tokens.extend(sentence_tokens)
            current_chunk_length += len(sentence_tokens)

    # 마지막에 남아있는 청크 저장
    if current_chunk_tokens:
        chunks.append(tokenizer.decode(current_chunk_tokens))
    
    return chunks

def save_processing_summary(pdf_files, success_files, failure_files):
    """현재까지의 처리 결과를 로그 파일로 저장합니다."""
    log_file_path = os.path.join(config.OUTPUT_BASE_DIR, "processing_summary.log")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("--- PDF 처리 결과 요약 ---\n\n")
            f.write(f"총 처리 시도: {len(pdf_files)}개\n")
            f.write(f"성공: {len(success_files)}개\n")
            f.write(f"실패: {len(failure_files)}개\n")
            f.write("\n--- 성공 목록 ---\n")
            for file_path in sorted(success_files):
                f.write(f"{os.path.basename(file_path)}\n")
            f.write("\n--- 실패 목록 ---\n")
            for file_path in sorted(failure_files):
                f.write(f"{os.path.basename(file_path)}\n")
        print(f"✅ 처리 결과 요약 파일이 '{log_file_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"!!!!!! 로그 파일 저장 중 오류 발생 !!!!!!")
        traceback.print_exc()

def run_processing():
    """PDF 처리는 병렬(CPU), 임베딩은 일괄(GPU)로 실행하여 ChromaDB에 저장합니다."""
    print("--- 전체 PDF 처리 및 임베딩 프로세스를 시작합니다. ---")
    
    if not os.path.isdir(config.INPUT_DIR):
        print(f"오류: 입력 디렉토리 '{config.INPUT_DIR}'를 찾을 수 없습니다.")
        return
    try:
        os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
    except OSError as e:
        print(f"오류: ChromaDB 디렉토리 생성에 실패했습니다. {e}")
        return

    pdf_files = [os.path.join(config.INPUT_DIR, f) for f in os.listdir(config.INPUT_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"경고: '{config.INPUT_DIR}'에 PDF 파일이 없습니다.")
        return
        
    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")

    # --- 모델 및 ChromaDB 클라이언트 초기화 ---
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"사용 디바이스: {device.upper()}")
        model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)
        model.half()  # FP16 추론 활성화
        embedding_function = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL_NAME, model_kwargs={'device': device})
        
        db = Chroma(
            persist_directory=str(config.CHROMA_DB_PATH),
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"!!!!!! 모델 또는 ChromaDB 초기화 중 오류 발생 !!!!!!")
        traceback.print_exc()
        return

    # --- 파일 목록을 배치로 나누기 ---
    num_batches = math.ceil(len(pdf_files) / config.PDF_BATCH_SIZE)
    pdf_batches = [pdf_files[i:i + config.PDF_BATCH_SIZE] for i in range(0, len(pdf_files), config.PDF_BATCH_SIZE)]

    success_files = []
    failure_files = []
    total_docs_added = 0
    num_processes = max(1, multiprocessing.cpu_count() - 1)

    print(f"{num_processes}개의 프로세스를 사용하여 PDF 텍스트 추출을 시작합니다.")
    print(f"총 {num_batches}개의 배치로 나누어 처리합니다.")

    # --- 배치별 처리 루프 ---
    for i, batch_files in enumerate(tqdm(pdf_batches, desc="전체 배치 처리 진행도")):
        print(f"--- 배치 {i+1}/{num_batches} 처리 시작 ---")
        
        all_pages = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(batch_files), desc=f"배치 {i+1} PDF 파싱") as pbar:
                for path, pages_content in pool.imap_unordered(process_pdf_to_pages, batch_files):
                    if pages_content:
                        all_pages.extend(pages_content)
                        if path not in success_files:
                            success_files.append(path)
                    else:
                        if path not in failure_files:
                            failure_files.append(path)
                    pbar.update(1)
        
        if not all_pages:
            print("현재 배치에서 처리된 데이터가 없습니다. 다음 배치를 진행합니다.")
            continue

        print(f"현재 배치에서 {len(all_pages)}개의 페이지 수집 완료. 청킹 및 임베딩을 시작합니다.")
        
        try:
            all_chunks_with_meta = []
            for page_data in tqdm(all_pages, desc="페이지 청킹"):
                chunks = chunk_text_from_page(page_data, model.tokenizer)
                for chunk_idx, chunk_text in enumerate(chunks):
                    all_chunks_with_meta.append({
                        "text": chunk_text,
                        "meta": {"source": page_data['source'], "page": page_data['page'], "chunk_index": chunk_idx}
                    })

            if not all_chunks_with_meta:
                print(f"경고: 배치 {i+1}에서 유효한 텍스트 청크를 찾을 수 없습니다. 임베딩을 건너뜁니다.")
                continue

            # --- 임베딩 전 길이 제한 적용 ---
            # 모델의 max_length를 명시적으로 설정하여 경고를 방지합니다.
            # 이 설정이 가장 확실한 방법입니다.
            embeddings = model.encode(
                [item['text'] for item in all_chunks_with_meta],
                show_progress_bar=True,
                batch_size=128,
                max_seq_length=512
            ).tolist()

            for item_idx, item in enumerate(all_chunks_with_meta):
                item['embedding'] = embeddings[item_idx]
                item['meta']['chunk_id'] = str(uuid.uuid4())

            for item_idx in range(len(all_chunks_with_meta)):
                if item_idx > 0:
                    all_chunks_with_meta[item_idx]['meta']["prev_chunk_id"] = all_chunks_with_meta[item_idx-1]['meta']["chunk_id"]
                if item_idx < len(all_chunks_with_meta) - 1:
                    all_chunks_with_meta[item_idx]['meta']["next_chunk_id"] = all_chunks_with_meta[item_idx+1]['meta']["chunk_id"]

            documents = [Document(page_content=item['text'], metadata=item['meta']) for item in all_chunks_with_meta]

            # --- ChromaDB 업로드 시 배치 제한 ---
            BATCH_LIMIT = 5000
            for start in range(0, len(documents), BATCH_LIMIT):
                db.add_documents(documents=documents[start:start+BATCH_LIMIT])
            
            total_docs_added += len(documents)
            print(f"✅ 배치 {i+1} 처리 완료. {len(documents)}개의 문서를 DB에 추가했습니다.")

        except Exception as e:
            print(f"!!!!!! 배치 {i+1} 임베딩 또는 DB 저장 중 오류 발생 !!!!!!")
            traceback.print_exc()

    # --- 최종 저장 및 정리 ---
    try:
        print("--- 모든 배치 처리 완료. 최종 DB 저장을 시작합니다. ---")
        db.persist()
        print(f"✅ 성공적으로 총 {total_docs_added}개의 문서를 ChromaDB에 저장했습니다.")
    except Exception as e:
        print(f"!!!!!! 최종 ChromaDB 저장 중 오류 발생 !!!!!!")
        traceback.print_exc()

    # 이 부분이 모든 작업이 완료된 후 실행됩니다.
    log_file_path = os.path.join(config.OUTPUT_BASE_DIR, "processing_summary.log")
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("--- PDF 처리 결과 요약 ---\n\n")
            f.write(f"총 처리 시도: {len(pdf_files)}개\n")
            f.write(f"성공: {len(success_files)}개\n")
            f.write(f"실패: {len(failure_files)}개\n")
            f.write("\n--- 성공 목록 ---\n")
            for file_path in sorted(success_files):
                f.write(f"{os.path.basename(file_path)}\n")
            f.write("\n--- 실패 목록 ---\n")
            for file_path in sorted(failure_files):
                f.write(f"{os.path.basename(file_path)}\n")
        print(f"✅ 처리 결과 요약 파일이 '{log_file_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"!!!!!! 로그 파일 저장 중 오류 발생 !!!!!!")
        traceback.print_exc()

    print("--- 모든 작업이 완료되었습니다. ---")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run_processing()

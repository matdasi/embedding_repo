import re

def clean_linebreaks(text: str) -> str:
    """하이픈으로 연결된 줄바꿈을 제거하고, 문장 중간의 줄바꿈을 공백으로 바꿉니다."""
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<![\.\?\!])\n(?=[^\n])', ' ', text)
    return text

def fix_numbering_patterns(text: str) -> str:
    """'1·', '01 .' 같은 다양한 번호 매기기 형식을 '1. '으로 통일합니다."""
    return re.sub(r'(\d{1,2})\s*[·\.]\s*', r'\1. ', text)

def fix_broken_numbers(text: str) -> str:
    """숫자와 기호(소수점, 퍼센트) 사이의 불필요한 공백을 제거합니다."""
    text = re.sub(r'\s*\.\s*', '.', text)
    text = re.sub(r'\s*%\s*', '%', text)
    return text

def remove_dot_leaders(text: str) -> str:
    """목차 등에서 사용되는 '....' 같은 점선을 공백으로 바꿉니다."""
    return re.sub(r'[·•\-\.]{2,}', ' ', text)

def clean_misparsed_headings(text: str) -> str:
    """'### 제목' 같은 마크다운 형식의 제목 기호를 제거합니다."""
    return re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

def remove_page_numbers(text: str) -> str:
    """'- 1 -', '51 -'과 같이 쪽 번호만 있는 줄을 제거합니다."""
    lines = []
    for ln in text.splitlines():
        if re.match(r'^\s*[-–]\s*\d+\s*[-–]\s*$', ln) or re.match(r'^\s*\d+\s*[-–]\s*$', ln):
            continue
        lines.append(ln)
    return '\n'.join(lines)

def is_useless_page(text: str) -> bool:
    """내용이 너무 짧거나, 글자보다 기호가 너무 많은 페이지를 건너뛸지 판단합니다."""
    if len(text) < 30:
        return True
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    # 기호의 비율이 70% 이상이면 의미 없는 페이지로 간주
    return symbols / max(len(text), 1) > 0.7

def is_text_corrupted(text: str, threshold: float = 0.05) -> bool:
    """텍스트에서 유니코드 대체 문자(U+FFFD)의 비율을 확인하여 깨짐 현상을 감지합니다."""
    if not text or len(text) < 20: # 텍스트가 너무 짧으면 판단에서 제외
        return False
    replacement_char_count = text.count('�')
    if replacement_char_count == 0:
        return False
    # 깨진 문자의 비율이 임계값을 넘으면 손상된 것으로 판단
    return (replacement_char_count / len(text)) > threshold



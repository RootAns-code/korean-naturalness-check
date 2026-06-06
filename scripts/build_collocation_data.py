#!/usr/bin/env python3
"""
명사+동사 결합 빈도 DB를 1회 빌드한다 (개발자용 스크립트).

Leipzig Korean kor_news_2022_1M 패키지에서 사전 계산된 좌/우 인접 공기
빈도 표(co_n.txt)를 받아 kiwipiepy로 형태소 분석한 뒤, 명사+동사 어간
결합만 추출해 sqlite db로 저장한다.

결과 DB: ../assets/kor_collocation.db (1M 기준 약 4MB)

이 스크립트는 스킬 사용자가 실행할 필요가 없다. 스킬 개발자가 한 번 빌드
해 결과 DB를 스킬 패키지에 동봉하면 모든 사용자(Claude.ai 웹, PC Claude
Code 모두)가 셋업 없이 즉시 사용할 수 있다.

사용법:
  python3 build_collocation_data.py

출처: Leipzig Corpora Collection - Korean (CC BY 4.0)
  https://wortschatz.uni-leipzig.de/en/download/Korean
"""
import os
import sqlite3
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

try:
    from kiwipiepy import Kiwi
except ImportError:
    print('FAIL: kiwipiepy 미설치. pip install kiwipiepy 후 재실행.')
    sys.exit(2)


URL = "https://downloads.wortschatz-leipzig.de/corpora/kor_news_2022_1M.tar.gz"
SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR / "_build_work"
OUTPUT_DB = SCRIPT_DIR.parent / "assets" / "kor_collocation.db"

# 필터링 임계값. 너무 희소한 페어는 잡음이라 제외.
MIN_FREQ = 2          # 페어가 최소 2회 이상 등장해야 유지
MIN_NOUN_LEN = 1      # 명사 어간 최소 길이 (1자 한자도 의미 있는 경우 있음)
MIN_VERB_LEN = 1      # 동사 어간 최소 길이


def download_corpus():
    WORK_DIR.mkdir(exist_ok=True)
    tar_path = WORK_DIR / Path(URL).name
    if tar_path.exists() and tar_path.stat().st_size > 1_000_000:
        print(f'  다운로드 캐시 사용: {tar_path} ({tar_path.stat().st_size//1024//1024} MB)')
        return tar_path
    print(f'  다운로드: {URL}')
    t0 = time.time()
    urllib.request.urlretrieve(URL, tar_path)
    elapsed = time.time() - t0
    size_mb = tar_path.stat().st_size // 1024 // 1024
    print(f'  완료: {size_mb} MB / {elapsed:.1f}s')
    return tar_path


def extract_corpus(tar_path):
    extract_dir = WORK_DIR / "extracted"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(exist_ok=True)
        print(f'  압축 해제: {tar_path.name}')
        with tarfile.open(tar_path) as t:
            t.extractall(extract_dir)
        print(f'  완료')
    return extract_dir


def find_co_n_file(extract_dir):
    candidates = list(extract_dir.rglob("*-co_n.txt"))
    if not candidates:
        raise FileNotFoundError("co_n.txt 파일을 찾을 수 없음")
    return candidates[0]


def find_words_file(extract_dir):
    candidates = list(extract_dir.rglob("*-words.txt"))
    if not candidates:
        raise FileNotFoundError("words.txt 파일을 찾을 수 없음")
    return candidates[0]


def load_word_map(words_path):
    """words.txt를 읽어 {word_id: word} 매핑 생성.

    형식: word_id TAB word TAB freq (3컬럼)
    """
    word_map = {}
    with open(words_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            try:
                word_id = int(parts[0])
                word = parts[1]
            except (ValueError, IndexError):
                continue
            word_map[word_id] = word
    return word_map


def normalize_word(kiwi, word, want_pos):
    """word를 형태소 분석해 want_pos(N 또는 V)에 해당하는 어간 반환.

    한국어는 어절 단위 결합이 활용형 다양성 때문에 희소해지므로 어간으로
    정규화해 같은 의미 동사의 활용형을 한 표제어로 통합한다.

    want_pos='N': 일반명사(NNG) 또는 고유명사(NNP) 첫 출현 형태 반환
    want_pos='V': 동사(VV) 또는 형용사(VA) 어간 반환
    """
    if not word or len(word) > 30:
        return None
    try:
        tokens = kiwi.tokenize(word)
    except Exception:
        return None
    if not tokens:
        return None
    if want_pos == 'N':
        # 명사 어절은 보통 명사+조사 구조. 첫 명사 추출.
        for tok in tokens:
            if tok.tag in ('NNG', 'NNP') and len(tok.form) >= MIN_NOUN_LEN:
                return tok.form
    elif want_pos == 'V':
        # 동사 어절은 동사어간+어미 구조. 첫 동사 어간 추출.
        for tok in tokens:
            if tok.tag in ('VV', 'VA') and len(tok.form) >= MIN_VERB_LEN:
                return tok.form
    return None


def parse_co_n(co_n_path, word_map, kiwi):
    """co_n.txt를 읽어 (noun, verb) -> (freq, sig) 누적 dict 반환.

    co_n.txt 형식: line_no TAB word_a_id TAB word_b_id TAB freq TAB sig
    word_map: {word_id: word_form} 매핑.
    """
    pair_freq = {}
    line_count = 0
    matched = 0
    # 형태소 분석 캐시 (같은 단어 반복 호출 회피)
    norm_cache_n = {}  # word -> noun 어간 or None
    norm_cache_v = {}  # word -> verb 어간 or None

    def cached_normalize(word, want_pos):
        cache = norm_cache_n if want_pos == 'N' else norm_cache_v
        if word in cache:
            return cache[word]
        result = normalize_word(kiwi, word, want_pos)
        cache[word] = result
        return result

    t0 = time.time()
    with open(co_n_path, encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count % 50000 == 0:
                elapsed = time.time() - t0
                print(f'    진행: {line_count}행 처리, 매칭 페어 누적 {matched}, 고유 페어 {len(pair_freq)}, {elapsed:.0f}s')
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            try:
                word_a_id = int(parts[0])
                word_b_id = int(parts[1])
                freq = int(parts[2])
                sig = float(parts[3]) if len(parts) >= 4 else 0.0
            except (ValueError, IndexError):
                continue

            w_a = word_map.get(word_a_id)
            w_b = word_map.get(word_b_id)
            if w_a is None or w_b is None:
                continue

            # word_a가 명사, word_b가 동사 시도
            noun = cached_normalize(w_a, 'N')
            verb = cached_normalize(w_b, 'V')
            if noun is None or verb is None:
                # 역순도 시도 (명사가 우측에 올 수 있음)
                noun = cached_normalize(w_b, 'N')
                verb = cached_normalize(w_a, 'V')
                if noun is None or verb is None:
                    continue

            key = (noun, verb)
            if key in pair_freq:
                cur_freq, cur_sig = pair_freq[key]
                pair_freq[key] = (cur_freq + freq, max(cur_sig, sig))
            else:
                pair_freq[key] = (freq, sig)
            matched += 1
    elapsed = time.time() - t0
    print(f'    완료: {line_count}행 전체, 매칭 페어 누적 {matched}, 고유 페어 {len(pair_freq)}, {elapsed:.0f}s')
    return pair_freq


def build_db(pair_freq):
    OUTPUT_DB.parent.mkdir(exist_ok=True, parents=True)
    if OUTPUT_DB.exists():
        OUTPUT_DB.unlink()

    conn = sqlite3.connect(OUTPUT_DB)
    conn.execute('''CREATE TABLE collocation (
        noun TEXT NOT NULL,
        verb TEXT NOT NULL,
        freq INTEGER NOT NULL,
        sig REAL NOT NULL
    )''')

    rows = [(n, v, f, s) for (n, v), (f, s) in pair_freq.items() if f >= MIN_FREQ]
    rows.sort(key=lambda r: -r[2])  # 빈도 내림차순

    conn.executemany("INSERT INTO collocation VALUES (?, ?, ?, ?)", rows)
    conn.execute("CREATE INDEX idx_noun_verb ON collocation(noun, verb)")
    conn.execute("CREATE INDEX idx_noun_sig ON collocation(noun, sig DESC)")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    size_kb = OUTPUT_DB.stat().st_size // 1024
    print(f'  빌드 완료: {OUTPUT_DB}')
    print(f'  페어 수: {len(rows)}')
    print(f'  DB 크기: {size_kb} KB')


def smoke_test():
    """빌드 결과 검증. 자연 페어와 어색 페어 빈도 차이 확인."""
    print()
    print('스모크 테스트:')
    conn = sqlite3.connect(OUTPUT_DB)
    test_pairs = [
        ('기업', '묶'),       # 어색 의심
        ('기업', '정리'),     # 자연
        ('기업', '소개'),     # 자연
        ('사례', '묶'),       # 어색 의심
        ('사례', '모으'),     # 자연
        ('기업', '참가'),     # 자연 (행사 맥락)
    ]
    for noun, verb in test_pairs:
        row = conn.execute(
            "SELECT freq, sig FROM collocation WHERE noun=? AND verb=?",
            (noun, verb)
        ).fetchone()
        if row:
            print(f'  ({noun}, {verb}): freq={row[0]}, sig={row[1]:.1f}')
        else:
            print(f'  ({noun}, {verb}): 0건 (코퍼스에 결합 없음)')
    print()
    # 명사 "기업"의 자연 동사 상위 5개
    print('  "기업" 명사의 자연 동사 상위 5개 (sig 기준):')
    for row in conn.execute(
        "SELECT verb, freq, sig FROM collocation WHERE noun='기업' ORDER BY sig DESC LIMIT 5"
    ):
        print(f'    {row[0]} (freq={row[1]}, sig={row[2]:.1f})')
    conn.close()


def main():
    print('한국어 명사+동사 결합 빈도 DB 빌드')
    print('='*60)
    print('[1/4] Leipzig kor_news_2022_1M 패키지 다운로드')
    tar_path = download_corpus()
    print('[2/4] 압축 해제')
    extract_dir = extract_corpus(tar_path)
    co_n_path = find_co_n_file(extract_dir)
    words_path = find_words_file(extract_dir)
    print(f'  co_n.txt 위치: {co_n_path}')
    print(f'  words.txt 위치: {words_path}')
    print('[3/4] 형태소 분석 및 명사+동사 페어 추출')
    print('  words.txt 매핑 로드 중...')
    word_map = load_word_map(words_path)
    print(f'  단어 매핑: {len(word_map)}개')
    kiwi = Kiwi()
    pair_freq = parse_co_n(co_n_path, word_map, kiwi)
    print('[4/4] SQLite DB 빌드')
    build_db(pair_freq)
    smoke_test()
    print('='*60)
    print('완료. 결과 DB를 스킬 패키지의 assets/ 에 동봉하면 됨.')


if __name__ == '__main__':
    main()

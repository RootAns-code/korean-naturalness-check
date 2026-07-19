#!/usr/bin/env python3
"""명사+동사 결합 빈도 DB를 1회 빌드한다 (개발자용 스크립트).

Leipzig Korean kor_news_2022_1M 패키지의 원문 문장(*-sentences.txt) 100만
문장을 kiwipiepy로 형태소 분석하고, 런타임 검사기(check_naturalness.py)의
페어 추출기를 **그대로 임포트**해 (명사, 동사 어간) 결합 빈도를 집계한 뒤
로그우도(LLR) 유의도와 함께 sqlite db로 저장한다.

v2까지는 Leipzig의 사전 계산 인접 공기표(co_n.txt)를 소비했는데, 인접 어절
공기만 담겨 있어 "사진을 예쁘게 찍다"처럼 수식어가 끼면 자연 결합도 0건이
되는 구조적 거짓양성이 있었다. 문장 원문 파이프라인은 런타임과 동일한
추출기로 동일한 거리 규칙의 페어를 집계하므로 DB-런타임 분포가 정의상
일치한다(2.1.3에서 확인된 추출-DB 불일치 버그 부류의 재발을 구조적으로 차단).

결과 DB: ../assets/kor_collocation.db

이 스크립트는 스킬 사용자가 실행할 필요가 없다. 스킬 개발자가 한 번 빌드
해 결과 DB를 스킬 패키지에 동봉하면 모든 사용자(Claude.ai 웹, PC Claude
Code 모두)가 셋업 없이 즉시 사용할 수 있다.

사용법:
  python3 build_collocation_data.py

출처: Leipzig Corpora Collection - Korean (CC BY 4.0)
  https://wortschatz.uni-leipzig.de/en/download/Korean
"""
import math
import os
import sqlite3
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    # 런타임과 동일한 추출기·형태소 분석기를 써야 DB와 런타임 분포가 일치한다.
    # check_naturalness의 추출 규칙(VA+게 스킵, "N과 같이" 스킵, 거리 10)을
    # 바꾸면 반드시 이 스크립트로 DB를 재빌드해야 한다.
    import check_naturalness as _cn
    from check_naturalness import _extract_noun_verb_pairs, _kiwi
    import kiwipiepy
except ImportError as e:
    print(f'FAIL: 의존성 로드 실패 ({e}). pip install kiwipiepy 후 재실행.')
    sys.exit(2)


URL = "https://downloads.wortschatz-leipzig.de/corpora/kor_news_2022_1M.tar.gz"
WORK_DIR = SCRIPT_DIR / "_build_work"
OUTPUT_DB = SCRIPT_DIR.parent / "assets" / "kor_collocation.db"

# 필터링 임계값. 너무 희소한 페어는 잡음이라 제외.
MIN_FREQ = 2          # 페어가 최소 2회 이상 등장해야 유지


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


def find_sentences_file(extract_dir):
    candidates = list(extract_dir.rglob("*-sentences.txt"))
    if not candidates:
        raise FileNotFoundError("sentences.txt 파일을 찾을 수 없음")
    return candidates[0]


def iter_sentences(sentences_path):
    """sentences.txt(형식: id TAB 문장)에서 문장만 순회."""
    with open(sentences_path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) < 2:
                continue
            sent = parts[1].strip()
            if sent:
                yield sent


def count_pairs(sentences_path):
    """전체 문장을 형태소 분석해 (명사, 동사) 페어 빈도와 주변 빈도를 집계."""
    pair_freq = Counter()
    noun_freq = Counter()
    verb_freq = Counter()
    n_sent = 0
    t0 = time.time()
    # kiwi.tokenize에 이터러블을 넘기면 내부 배치 처리로 문장별 토큰열을 돌려준다
    for tokens in _kiwi.tokenize(iter_sentences(sentences_path)):
        n_sent += 1
        if n_sent % 100000 == 0:
            elapsed = time.time() - t0
            print(f'    진행: {n_sent}문장, 고유 페어 {len(pair_freq)}, {elapsed:.0f}s')
        for noun, verb in _extract_noun_verb_pairs(tokens):
            pair_freq[(noun, verb)] += 1
            noun_freq[noun] += 1
            verb_freq[verb] += 1
    elapsed = time.time() - t0
    print(f'    완료: {n_sent}문장, 고유 페어 {len(pair_freq)}, {elapsed:.0f}s')
    return pair_freq, noun_freq, verb_freq


def _ll(x):
    return x * math.log(x) if x > 0 else 0.0


def log_likelihood(k11, noun_total, verb_total, grand_total):
    """Dunning 로그우도비. 값이 클수록 우연이 아닌 잘 입증된 결합."""
    k12 = noun_total - k11
    k21 = verb_total - k11
    k22 = grand_total - k11 - k12 - k21
    if k22 < 0:
        k22 = 0
    r1, r2 = k11 + k12, k21 + k22
    c1, c2 = k11 + k21, k12 + k22
    llr = 2.0 * (
        _ll(k11) + _ll(k12) + _ll(k21) + _ll(k22)
        - _ll(r1) - _ll(r2) - _ll(c1) - _ll(c2)
        + _ll(r1 + r2)
    )
    return max(llr, 0.0)


def build_db(pair_freq, noun_freq, verb_freq):
    OUTPUT_DB.parent.mkdir(exist_ok=True, parents=True)
    # check_naturalness 임포트가 현행 DB에 연결을 연 상태라 Windows에서는
    # 바로 덮어쓸 수 없다. 임시 파일에 빌드한 뒤 연결을 닫고 원자 교체한다.
    tmp_db = OUTPUT_DB.with_name(OUTPUT_DB.name + '.new')
    if tmp_db.exists():
        tmp_db.unlink()

    grand_total = sum(pair_freq.values())
    rows = []
    for (noun, verb), freq in pair_freq.items():
        if freq < MIN_FREQ:
            continue
        sig = log_likelihood(freq, noun_freq[noun], verb_freq[verb], grand_total)
        rows.append((noun, verb, freq, sig))
    rows.sort(key=lambda r: -r[2])  # 빈도 내림차순

    conn = sqlite3.connect(tmp_db)
    conn.execute('''CREATE TABLE collocation (
        noun TEXT NOT NULL,
        verb TEXT NOT NULL,
        freq INTEGER NOT NULL,
        sig REAL NOT NULL
    )''')
    conn.executemany("INSERT INTO collocation VALUES (?, ?, ?, ?)", rows)
    conn.execute("CREATE INDEX idx_noun_verb ON collocation(noun, verb)")
    conn.execute("CREATE INDEX idx_noun_sig ON collocation(noun, sig DESC)")
    # 빌드 출처 메타데이터. 런타임 필수 아님(향후 kiwi 버전 불일치 진단용).
    conn.execute('CREATE TABLE build_info (key TEXT PRIMARY KEY, value TEXT)')
    conn.executemany("INSERT INTO build_info VALUES (?, ?)", [
        ('corpus', Path(URL).name),
        ('pipeline', 'sentences-runtime-extractor'),
        ('kiwipiepy_version', kiwipiepy.__version__),
        ('built_date', date.today().isoformat()),
        ('pair_instances_total', str(grand_total)),
        ('min_freq', str(MIN_FREQ)),
    ])
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    if _cn._collocation_conn is not None:
        _cn._collocation_conn.close()
        _cn._collocation_conn = None
    os.replace(tmp_db, OUTPUT_DB)

    size_kb = OUTPUT_DB.stat().st_size // 1024
    print(f'  빌드 완료: {OUTPUT_DB}')
    print(f'  페어 수: {len(rows)} (인스턴스 {grand_total})')
    print(f'  DB 크기: {size_kb} KB')


def smoke_test():
    """빌드 결과 검증. 자연 페어와 어색 페어 빈도 차이 확인."""
    print()
    print('스모크 테스트:')
    conn = sqlite3.connect(OUTPUT_DB)
    test_pairs = [
        ('기업', '묶이'),     # 어색 의심 (LLM 산출물에서 관찰)
        ('기업', '소개'),     # 자연
        ('사례', '모으'),     # 자연
        ('사진', '찍'),       # 자연 (비인접 수식 구문에서도 잡혀야 함)
        ('빵', '사'),         # 자연 (구 co_n DB에서 0건이던 일상 결합)
        ('인사', '건네'),     # 자연
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
    print('  "사진" 명사의 자연 동사 상위 5개 (sig 기준):')
    for row in conn.execute(
        "SELECT verb, freq, sig FROM collocation WHERE noun='사진' ORDER BY sig DESC LIMIT 5"
    ):
        print(f'    {row[0]} (freq={row[1]}, sig={row[2]:.1f})')
    conn.close()


def main():
    print('한국어 명사+동사 결합 빈도 DB 빌드 (문장 원문 파이프라인)')
    print('='*60)
    print('[1/4] Leipzig kor_news_2022_1M 패키지 다운로드')
    tar_path = download_corpus()
    print('[2/4] 압축 해제')
    extract_dir = extract_corpus(tar_path)
    sentences_path = find_sentences_file(extract_dir)
    print(f'  sentences.txt 위치: {sentences_path}')
    print('[3/4] 형태소 분석 및 명사+동사 페어 집계')
    pair_freq, noun_freq, verb_freq = count_pairs(sentences_path)
    print('[4/4] SQLite DB 빌드 (LLR 유의도 계산 포함)')
    build_db(pair_freq, noun_freq, verb_freq)
    smoke_test()
    print('='*60)
    print('완료. 결과 DB를 스킬 패키지의 assets/ 에 동봉하면 됨.')


if __name__ == '__main__':
    main()

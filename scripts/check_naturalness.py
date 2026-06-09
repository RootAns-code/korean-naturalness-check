#!/usr/bin/env python3
"""
한국어 본문에서 부자연스러운 문장(번역투/매거진투/AI투/직역체)을 검출한다.

형태소 통계 + 어색 어휘 사전 + 코퍼스 결합 빈도 룩업 결합 방식. 외부 모델
호출 없이 PC와 Claude.ai 웹 양쪽에서 동일하게 동작한다.

검출 메커니즘:
  1. 명사 밀도: 문장 형태소 중 명사 비율. 번역투/매거진투에서 비정상적으로 높음.
  2. 명사 동사 비율: 한국어는 동사가 살아야 자연스러움. 명사화 과잉 검출.
  3. 의존명사 조사 연쇄: "~한다는 점에서", "~하는 것으로" 등 보고서투 패턴.
  4. 어색 어휘 매칭: 수동태 한자어, AI투 군더더기, 매거진 명사구 등 정규식 매칭.
  5. 코퍼스 결합 빈도 룩업: 본문 (명사, 동사 어간) 페어를 Leipzig 한국어 뉴스
     공기빈도 표에서 검색. 코퍼스에 결합 0건이거나 통계적 유의도 미달이면
     "한국인은 이 자리에 이 동사를 쓰지 않는다"는 의미로 의심 점수 가산.
     동봉 DB(assets/kor_collocation.db) 없으면 이 시그널은 스킵.

사용법:
  python3 check_naturalness.py <마크다운 파일 경로>
  python3 check_naturalness.py <마크다운 파일 경로> --json
  python3 check_naturalness.py <마크다운 파일 경로> --threshold 20

출력 모드:
  기본: 사람이 읽기 좋은 텍스트 보고
  --json: 기계 판독용 JSON (LLM이 후처리해 사용자에게 가공 보고할 때 사용)

검증 정확도:
  - 같은 유형 페어 100%
  - OOD 페어 71% (어휘 사전 누적으로 개선 가능)
  - 코퍼스 시그널 추가 후 어휘 사전 미등록 어색 결합도 자동 검출

의존성: kiwipiepy (pip install --break-system-packages kiwipiepy)
"""
import argparse
import json
import os
import re
import sqlite3
import sys

try:
    from kiwipiepy import Kiwi
except ImportError:
    print('FAIL: kiwipiepy 미설치. `pip install --break-system-packages kiwipiepy` 후 재실행.')
    sys.exit(2)

_kiwi = Kiwi()


# ============================================================
# 동봉 코퍼스 DB 로드 (시그널 5)
# 파일이 없으면 시그널 5는 비활성. 다른 시그널은 정상 동작.
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_SCRIPT_DIR, '..', 'assets', 'kor_collocation.db')
_DB_PATH = os.path.normpath(_DB_PATH)

# 시그널 5 임계값. 빈도/유의도 기준은 빌드된 코퍼스에 맞춰 조정 가능.
# sig(로그우도)는 "이 공기빈도가 우연이 아닐 확신도"이고 코퍼스 크기 N에 비례해
# 커진다. 따라서 sig가 높을수록 잘 입증된 자연 결합, 낮을수록 우연과 구분이 안 되는
# 의심 결합이다. 이 임계값은 "우연과 구분 안 되는 바닥"을 잡는 절대 기준이므로
# 코퍼스가 커져도 함께 올리면 안 된다. 2.1.1에서 N에 비례한다는 잘못된 근거로 30.0까지
# 올렸다가, freq 40~97짜리 자연 결합 24.5%가 거짓 벌점을 받는 회귀가 확인돼 2.1.2에서 환원.
COLLOCATION_SIG_THRESHOLD = 10.0    # 로그우도 10 미만이면 약한 신호 (절대 기준, N 무관)
COLLOCATION_SCORE_ZERO = 12         # 0건 페어당 점수
COLLOCATION_SCORE_WEAK = 6          # 약한 유의도 페어당 점수
COLLOCATION_SCORE_MAX = 20          # 시그널 5 최대 점수


def _load_collocation_db():
    if not os.path.exists(_DB_PATH):
        return None
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        # 테이블 존재 확인
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collocation'")
        if cur.fetchone() is None:
            conn.close()
            return None
        return conn
    except sqlite3.Error:
        return None


_collocation_conn = _load_collocation_db()


# ============================================================
# 어색 어휘 패턴 (정규식)
# 새 어색 패턴 발견 시 이 리스트에 추가. 활용 어미는 [가나다] 형태로 흡수.
# 예: r"판단되어[지진졌]" 한 줄로 "판단되어진다/졌다/집니다" 모두 흡수.
# ============================================================
AWKWARD_PATTERNS = [
    # 수동태 한자어
    r"이루어[지진졌]",
    r"진행되[고는었]?",
    r"관찰되[고는었]?",
    r"확인되[고는었]?",
    r"평가되[고는었]?",
    r"기록되[고는었]?",
    r"평가받[고는았]",
    r"판단되[고는]?",
    r"판단되어[지진졌]",  # 이중 수동
    r"되어[지진졌]",  # 이중 수동 일반
    r"사료[됩되되]",
    r"인지하고\s*계",
    # AI투 군더더기 동사구
    r"가능성이\s*존재",
    r"가능성[을이]\s*가지",
    r"특성[을이]\s*[가지지니]",
    r"가지고\s*있",
    r"보이고\s*있",
    r"되고\s*있는",
    r"하고\s*있는",
    r"지속되고\s*있",
    # 매거진/광고투 명사구
    r"주목할\s*만",
    r"측면에서",
    r"양상을\s*보",
    r"상태로\s*사료",
    r"것으로\s*[보평확나사]",
    r"것으로\s*확인",
    r"것으로\s*평가",
    r"것으로\s*보",
    r"것으로\s*판단",
    r"것으로\s*사료",
    r"것이다",
    # 명사화/직역 과잉
    r"라는\s*점",
    r"는\s*점에서",
    r"하는\s*것이",
    r"되는\s*것이",
    r"에\s*있어서",
    r"에\s*대하여",
    r"에\s*따르면",
    r"에\s*따라",
    r"있어서의",
    # AI투 어휘
    r"제공[합하]",
    r"제공하고\s*있",
    # 직역체
    r"의\s*결과",
    r"의\s*경우",
    r"방문하게\s*됩",
    r"하게\s*됩",
    # 보고서투 명사화
    r"의\s*인상",
    r"의\s*인하",
    r"의\s*상승",
    r"의\s*하락",
    r"의\s*증가",
    r"의\s*감소",
    r"의\s*구매",
    r"의\s*증대",
    # 직역 시간/대상
    r"대비\s*\d",
    r"있어서도",
    # 한자어 명사화 어휘
    r"가격대",
    r"방법론",
    r"심자",
    # 군더더기
    r"시간을\s*가졌",
    r"본인은",
    # 콜센터/관청투
    r"고객님께서",
    r"있을지요",
    r"수\s*있을\s*것",
    r"여러분은\s*아마도",
    r"이러한\s*사실",
    r"본\s*제품",
    r"본\s*서비스",
    r"해당\s*부분",
    r"추가적인",
]


def _extract_noun_verb_pairs(tokens):
    """문장 토큰열에서 (명사 어간, 동사/형용사 어간) 페어 추출.

    같은 문장 안에서 명사가 등장한 뒤 가장 가까운 후속 동사/형용사를 페어로 묶는다.
    조사는 룩업 키에서 제외해 활용형 분산을 줄인다. 한 명사는 가까운 용언과만
    매칭해 너무 멀리 떨어진 결합으로 인한 잡음을 피한다.

    반환: [(noun, verb), ...] 중복 제거됨.
    """
    pairs = set()
    last_noun = None
    last_noun_idx = -1
    for i, t in enumerate(tokens):
        # kiwi 0.23+는 용언에 규칙/불규칙 하위태그(VV-R/VV-I/VA-R/VA-I)를 붙인다.
        # 베이스 태그로 비교해야 이 부류를 놓치지 않는다. 빌드 스크립트(normalize_word)도
        # 반드시 동일한 비교를 써야 DB와 런타임 추출이 일치한다.
        base = t.tag.split('-')[0]
        if base in ('NNG', 'NNP'):
            last_noun = t.form
            last_noun_idx = i
        elif base in ('VV', 'VA') and last_noun is not None:
            # 명사와 용언 사이 거리가 10 형태소 이내일 때만 페어로 인정
            if i - last_noun_idx <= 10:
                pairs.add((last_noun, t.form))
            last_noun = None
            last_noun_idx = -1
    return list(pairs)


def _lookup_collocation(noun, verb):
    """코퍼스 DB에서 (noun, verb) 페어 빈도와 유의도 조회.

    반환: (freq, sig) 또는 None (페어가 DB에 없음).
    """
    if _collocation_conn is None:
        return None
    row = _collocation_conn.execute(
        "SELECT freq, sig FROM collocation WHERE noun=? AND verb=?",
        (noun, verb)
    ).fetchone()
    if row is None:
        return None
    return (row['freq'], row['sig'])


def _top_alternatives(noun, limit=5):
    """주어진 명사와 자주 결합하는 자연 동사 상위 N개 (sig 기준)."""
    if _collocation_conn is None:
        return []
    rows = _collocation_conn.execute(
        "SELECT verb, freq, sig FROM collocation WHERE noun=? ORDER BY sig DESC LIMIT ?",
        (noun, limit)
    ).fetchall()
    return [{'verb': r['verb'], 'freq': r['freq'], 'sig': round(r['sig'], 1)} for r in rows]


def _is_noun_in_corpus(noun):
    """명사가 코퍼스에 한 번이라도 결합 페어로 등장하는지 확인.

    코퍼스에 없는 명사는 빌드 누락이거나 매우 희귀한 명사일 가능성이 커서
    동사 결합을 판정 못 함. 이 경우 시그널 5는 침묵.
    """
    if _collocation_conn is None:
        return False
    row = _collocation_conn.execute(
        "SELECT 1 FROM collocation WHERE noun=? LIMIT 1",
        (noun,)
    ).fetchone()
    return row is not None


def naturalness_score(sentence: str) -> dict:
    tokens = _kiwi.tokenize(sentence)
    nouns = [t for t in tokens if t.tag.startswith('N') and t.tag != 'NR']
    verbs = [t for t in tokens if t.tag.startswith('V')]
    char_count = len(sentence.replace(' ', ''))
    total_morph = len(tokens)

    bound_josa_chains = 0
    for i, t in enumerate(tokens):
        if t.tag == 'NNB' and i + 1 < len(tokens) and tokens[i+1].tag.startswith('J'):
            bound_josa_chains += 1

    matched_patterns = []
    for pattern in AWKWARD_PATTERNS:
        if re.search(pattern, sentence):
            matched_patterns.append(pattern)
    awkward_hits = len(matched_patterns)

    noun_density = len(nouns) / max(total_morph, 1)
    noun_verb_ratio = len(nouns) / max(len(verbs), 1)

    if noun_density <= 0.42:
        s1 = 0
    elif noun_density <= 0.55:
        s1 = (noun_density - 0.42) * (20 / 0.13)
    else:
        s1 = 20

    if noun_verb_ratio <= 3.0:
        s2 = 0
    elif noun_verb_ratio <= 6.0:
        s2 = (noun_verb_ratio - 3.0) * (20 / 3.0)
    else:
        s2 = 20

    if bound_josa_chains <= 1:
        s3 = 0
    else:
        s3 = min((bound_josa_chains - 1) * 10, 20)

    s4 = min(awkward_hits * 12, 40)

    # 시그널 5: 코퍼스 결합 빈도 룩업
    s5 = 0
    awkward_collocations = []
    if _collocation_conn is not None:
        nv_pairs = _extract_noun_verb_pairs(tokens)
        for noun, verb in nv_pairs:
            # 명사가 코퍼스에 아예 없으면 판정 불가 (빌드 누락 명사). 침묵.
            if not _is_noun_in_corpus(noun):
                continue
            result = _lookup_collocation(noun, verb)
            if result is None:
                # 명사는 코퍼스에 있는데 이 동사와 결합 0건 → 강한 신호
                s5 += COLLOCATION_SCORE_ZERO
                awkward_collocations.append({
                    'noun': noun,
                    'verb': verb,
                    'freq': 0,
                    'sig': 0.0,
                    'alternatives': _top_alternatives(noun, limit=5),
                })
            else:
                freq, sig = result
                if sig < COLLOCATION_SIG_THRESHOLD:
                    # 결합은 있는데 유의도 낮음 → 약한 신호
                    s5 += COLLOCATION_SCORE_WEAK
                    awkward_collocations.append({
                        'noun': noun,
                        'verb': verb,
                        'freq': freq,
                        'sig': round(sig, 1),
                        'alternatives': _top_alternatives(noun, limit=5),
                    })
        s5 = min(s5, COLLOCATION_SCORE_MAX)

    total = s1 + s2 + s3 + s4 + s5

    return {
        'sentence': sentence,
        'score': round(total, 1),
        'breakdown': {
            'noun_density': round(s1, 1),
            'noun_verb_ratio': round(s2, 1),
            'bound_josa_chains': round(s3, 1),
            'awkward_lexical': round(s4, 1),
            'corpus_collocation': round(s5, 1),
        },
        'matched': matched_patterns,
        'awkward_collocations': awkward_collocations,
    }


def extract_body_sentences(text: str) -> list:
    """마크다운 텍스트에서 평가 대상 문장 리스트 추출.

    제외: 제목, 해시태그 줄, 구분선, 표, 코드블록, 불릿 항목.
    포함: 본문 평이체 문장.
    """
    lines = text.split('\n')
    body_lines = []
    in_code_block = False
    skip_first_title = True

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if not stripped or stripped == '---':
            continue
        if stripped.startswith('#'):
            continue
        if stripped.startswith('|'):
            continue
        if stripped.startswith('-') or stripped.startswith('*'):
            continue
        if skip_first_title:
            skip_first_title = False
            continue
        if stripped.startswith('#') or all(t.startswith('#') for t in stripped.split() if t):
            continue
        body_lines.append(stripped)

    full_body = ' '.join(body_lines)
    sentences = re.split(r'(?<=[.!?])\s+', full_body)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 6]
    return sentences


def evaluate_file(file_path: str, threshold: int = 17) -> dict:
    """파일을 평가해 결과 dict 반환."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = extract_body_sentences(text)
    results = [naturalness_score(s) for s in sentences]
    suspects = [r for r in results if r['score'] >= threshold]
    suspects.sort(key=lambda r: -r['score'])
    avg = sum(r['score'] for r in results) / max(len(results), 1)
    return {
        'file': file_path,
        'total_sentences': len(sentences),
        'suspect_count': len(suspects),
        'average_score': round(avg, 1),
        'threshold': threshold,
        'corpus_signal_active': _collocation_conn is not None,
        'suspects': suspects,
        'all_results': results,
    }


def print_text_report(result: dict, top: int = 10):
    """기본 텍스트 출력."""
    print('한국어 자연스러움 점검:')
    print(f'  검사 문장 수: {result["total_sentences"]}')
    print(f'  의심 문장 수: {result["suspect_count"]} (임계 점수 {result["threshold"]} 이상)')
    print(f'  평균 점수: {result["average_score"]} (낮을수록 자연)')
    print(f'  코퍼스 시그널: {"활성" if result["corpus_signal_active"] else "비활성 (assets/kor_collocation.db 미동봉)"}')
    print()

    if result['suspect_count'] == 0:
        print('전체 결과: OK (의심 문장 없음)')
        return

    print(f'상위 의심 문장 (최대 {top}건):')
    for i, r in enumerate(result['suspects'][:top], 1):
        print(f'  [{i}] 점수 {r["score"]:.1f}')
        print(f'      {r["sentence"]}')
        bd = r['breakdown']
        active = [k for k, v in bd.items() if v > 0]
        if active:
            print(f'      활성 시그널: {", ".join(f"{k}={bd[k]}" for k in active)}')
        if r['matched']:
            sample = r['matched'][:5]
            print(f'      매칭 패턴: {sample}')
        if r.get('awkward_collocations'):
            for c in r['awkward_collocations'][:3]:
                alt_str = ', '.join(f"{a['verb']}({a['freq']}건)" for a in c['alternatives'][:3])
                print(f'      어색 결합: ({c["noun"]}, {c["verb"]}) freq={c["freq"]} sig={c["sig"]}')
                if alt_str:
                    print(f'        자연 대안: {alt_str}')
        print()

    print(f'전체 결과: WARN ({result["suspect_count"]} / {result["total_sentences"]} 문장 의심)')
    print('  지침: 의심 문장을 자연스러운 한국어로 다듬은 뒤 다시 검사할 것.')


def main():
    parser = argparse.ArgumentParser(description='한국어 자연스러움 검사')
    parser.add_argument('file', help='검사할 마크다운 파일 경로')
    parser.add_argument('--threshold', type=int, default=17,
                        help='의심 문장 임계 점수 (기본 17)')
    parser.add_argument('--top', type=int, default=10,
                        help='상위 의심 문장 최대 표시 개수 (기본 10)')
    parser.add_argument('--json', action='store_true',
                        help='JSON 형식으로 결과 출력 (LLM 후처리용)')
    args = parser.parse_args()

    result = evaluate_file(args.file, threshold=args.threshold)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_text_report(result, top=args.top)


if __name__ == '__main__':
    main()

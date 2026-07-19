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
#
# 약한-신호 가지의 역사: sig가 낮다 = "공기빈도가 우연과 구분되지 않는다"인데,
# 이것은 어색함의 증거가 아니다. (생각,있)·(관심,있)처럼 두 단어가 모두 흔하면
# 자연 결합도 기대빈도와 관측빈도가 비슷해 sig가 낮게 나온다. 2.3.0에서 LLR을
# 자체 계산하게 되자 in-DB 페어의 55.6%가 sig<10으로 떨어져 이 가지가 자연
# 결합을 대량 벌점하는 회귀가 실측됐다(2.1.1 회귀와 동일 부류). 그래서 약한
# 벌점은 "빈도도 희소하고(sig 판정이 무의미한 수준) 유의도도 바닥"인 페어로
# 게이트한다. 조정 시 eval/run_eval.py로 전후 실측이 필수.
COLLOCATION_SIG_THRESHOLD = 3.84    # LLR p<0.05 경계. 이 미만이면 우연과 구분 불가
COLLOCATION_WEAK_MAX_FREQ = 4       # 이 빈도 이하 + 저유의도일 때만 약한 벌점
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
# 어색 어휘 패턴 그룹 (정규식)
#
# 같은 현상을 겨누는 패턴들을 한 그룹으로 묶고, 그룹당 1회만 가산한다.
# ("것으로 평가되고" 하나가 패턴 3개에 겹쳐 36점을 받던 중복 과금 방지)
#
# weight 18 = 확실 신호: 자연 한국어에서 사실상 안 나오는 문형("되어집니다",
#             "~에 있어서"). 단독으로 임계(17)를 넘어 즉시 의심 분류.
# weight 12 = 강한 신호: 어색 경향이 강하나 격식문 자연 용례도 있는 문형
#             ("확인되었다"). 다른 신호 하나와 합쳐지면 임계를 넘는다.
# weight 6  = 약한 신호: 자연 용례도 흔한 문체 경향("의 경우", "것이다", "에 따라").
#             단독·2연타로는 임계 미달, 강한 신호의 보조 근거로만 작동.
# 티어 배치는 eval/run_eval.py 라벨셋으로 실측 검증됨. 조정 시 재실측 필수.
#
# 새 패턴 추가 시: 해당하는 그룹에 넣거나 새 그룹을 만든다. 활용 어미는
# [가나다] 문자클래스로 흡수하되, 융합 음절을 빠뜨리지 않도록 주의.
# 예: "되어지다"의 활용은 되어지/되어진/되어질/되어집(니다)/되어짐/되어졌으로
# 실현되므로 r"되어[지진질집짐졌]"이어야 "되어집니다"를 놓치지 않는다.
# ============================================================
AWKWARD_PATTERN_GROUPS = [
    # --- 확실 신호 (18점): 자연 한국어에서 사실상 안 나오는 문형. 단독으로 임계(17) 통과 ---
    ('이중수동', 18, [
        r"되어[지진질집짐졌져]",
        r"이루어[지진질집짐졌져]",
        r"받아[지진질집짐졌져]",
        r"모아[지진질집짐졌져]",
        r"보여[지진질집짐졌져]",
        r"쓰여[지진질집짐졌져]",
        r"지어지",
        r"사료[되됩됨]",
    ]),
    ('것으로_판단동사', 18, [
        r"것으로\s*[보평확나사판알전풀]",
    ]),
    ('번역투_있어서', 18, [
        r"에\s*있어서?",
        r"있어서의",
        r"있어서도",
        r"에\s*대하여",
    ]),
    ('매거진_지시어', 18, [
        r"본\s*(제품|서비스|리뷰|글|포스팅|상품|메뉴|업체)",
        r"해당\s*(부분|제품|사항|이슈|장소|메뉴|내용)",
    ]),
    ('보고서투_명사구', 18, [
        r"주목할\s*만",
        r"측면에서",
        r"양상을\s*보",
        r"는\s*점에서",
        r"이러한\s*사실",
        r"(당신|여러분)[은이]?\s*아마도",
        r"본인은",
        r"로\s*하여금",
    ]),
    ('존재_가능_번역투', 18, [
        r"(가능성|여지|필요성)[이가]\s*존재",
        r"[이가]\s*존재[하합해한]",
        r"(확인|구매|사용|이용|주문|획득|예매)[이가]\s*가능[하합해한]",
        r"함으로써",
    ]),
    ('경직_동사남용', 18, [
        r"[을를]\s*(소유|보유)[하한]",
        r"[을를]\s*수행[하했합한]",
        r"[을를]\s*영위[하했합한]",
        r"[을를]\s*경험[하했할합]",
        r"[을를]\s*획득[하했합한]",
        r"[을를]\s*진행[하했합한]",
        r"노력을\s*(기울이|경주)",
        r"[다라]고\s*(말|평가|표현)할\s*수",
        r"는\s*것을\s*(돕|지속|완료|사랑|멈추)",
    ]),
    ('콜센터_관청투', 18, [
        r"고객님께서",
        r"있을지요",
        r"인지하고\s*계",
    ]),
    ('수있을것', 18, [
        r"수\s*있을\s*것",
    ]),
    # --- 강한 신호 (12점): 어색 경향이 강하나 격식문에 자연 용례도 있는 문형 ---
    ('보고서투_명사화_의', 12, [
        r"의\s*(인상|인하|상승|하락|증가|감소|구매|증대|확보|유지|해소|실천|실시|획득|제공|향상|도출|달성|마무리|조화)",
    ]),
    ('가지다_남용', 12, [
        r"가지고\s*있",
        r"[을를]\s*가[지진질집짐졌져]",
        r"가능성[을이]\s*가지",
        r"특성[을이]\s*[가지지니]",
        r"시간을\s*가졌",
    ]),
    ('수동태_한자어', 12, [
        r"(진행|관찰|확인|평가|기록|판단|형성|도출|요구|권장|추천|예상|발생|지원|구성|작성|시행|생각)[되됩됨]",
    ]),
    # --- 약한 신호 (6점) ---
    ('문어체_조사구', 6, [
        r"의\s*경우",
        r"에\s*따라",
        r"에\s*따르면",
        r"의\s*결과",
        r"라는\s*점",
        r"대비\s*\d",
    ]),
    ('것이다_종결', 6, [
        r"것이다",
        r"할\s*것입니다",
    ]),
    ('진행형_남용', 6, [
        r"(되|보이|지)고\s*있",
        r"지속되고\s*있",
        r"제공하고\s*있",
    ]),
    ('제공_남용', 6, [
        r"제공[하합해한]",
    ]),
    ('명사절_직역', 6, [
        r"하는\s*것[이은]",
        r"되는\s*것[이은]",
        r"는\s*것을\s*(돕|지속|완료|사랑|멈추)",
    ]),
    ('한자어_적인', 6, [
        r"(추가|지속|효율|전반|체계|안정|전략|즉각|성공)적[인으]",
    ]),
    ('하게됨', 6, [
        r"하게\s*됩",
    ]),
    ('으로_작용', 6, [
        r"으로\s*작용[하해]",
    ]),
]

# 하위 호환: 평면 패턴 리스트가 필요한 외부 소비자용
AWKWARD_PATTERNS = [p for _, _, ps in AWKWARD_PATTERN_GROUPS for p in ps]


def _extract_noun_verb_pairs(tokens):
    """문장 토큰열에서 (명사 어간, 동사/형용사 어간) 페어 추출.

    같은 문장 안에서 명사가 등장한 뒤 가장 가까운 후속 동사/형용사를 페어로 묶는다.
    조사는 룩업 키에서 제외해 활용형 분산을 줄인다. 한 명사는 가까운 용언과만
    매칭해 너무 멀리 떨어진 결합으로 인한 잡음을 피한다.

    빌드 스크립트(build_collocation_data.py)가 이 함수를 그대로 임포트해 코퍼스를
    집계하므로, 여기를 고치면 DB도 재빌드해야 추출-DB가 일치한다(2.1.3 교훈).

    반환: [(noun, verb), ...] 중복 제거됨.
    """
    pairs = set()
    last_noun = None
    last_noun_idx = -1
    n = len(tokens)
    for i, t in enumerate(tokens):
        # kiwi 0.23+는 용언에 규칙/불규칙 하위태그(VV-R/VV-I/VA-R/VA-I)를 붙인다.
        # 베이스 태그로 비교해야 이 부류를 놓치지 않는다.
        base = t.tag.split('-')[0]
        if base in ('NNG', 'NNP'):
            # "N과 같이/같은"의 N은 비교 관용구라 서술 관계가 아님. 이 명사를
            # 건너뛰고 앞선 명사를 유지해야 "기업이 다음과 같이 묶입니다"에서
            # (다음, 묶이)가 아니라 (기업, 묶이)가 나온다. "N보다"(비교 부사어,
            # 예: "생각보다 재미있다")의 N도 서술 관계가 아니므로 같이 건너뛴다.
            nxt = tokens[i + 1] if i + 1 < n else None
            nx2 = tokens[i + 2] if i + 2 < n else None
            if (nxt is not None and nxt.tag.startswith('J') and nxt.form in ('과', '와')
                    and nx2 is not None and nx2.form.startswith('같')):
                continue
            if nxt is not None and nxt.tag.startswith('J') and nxt.form == '보다':
                continue
            last_noun = t.form
            last_noun_idx = i
        elif base in ('VV', 'VA') and last_noun is not None:
            # "사진을 예쁘게 찍다"의 '예쁘게'(VA+게)는 부사어라 명사의 서술어가
            # 아님. 페어를 만들지도, 명사를 소모하지도 않고 본용언을 기다린다.
            nxt = tokens[i + 1] if i + 1 < n else None
            if base == 'VA' and nxt is not None and nxt.form == '게' and nxt.tag == 'EC':
                continue
            # 지시 용언(그렇다/이렇다/어떻다)은 내용 서술어가 아니라 페어 무의미.
            # "노트북이라 그런지"가 (노트북, 그렇)로 잡히는 잡음 방지.
            if t.form in ('그렇', '이렇', '저렇', '어떻', '그러', '이러', '저러'):
                continue
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


# 대안 동사 추천에서 제외할 기능성 용언. 어떤 명사와도 결합해 sig 상위를
# 도배하지만("기업에 대하", "~를 위하") 윤문 대안으로는 정보 가치가 없다.
FUNCTION_VERBS = frozenset([
    '있', '없', '하', '되', '대하', '위하', '통하', '따르', '관하', '의하',
    '인하', '비롯하', '같', '이러하', '그러하', '아니하', '못하', '말미암',
])


def _top_alternatives(noun, limit=5):
    """주어진 명사와 자주 결합하는 자연 동사 상위 N개 (sig 기준).

    기능성 용언(FUNCTION_VERBS)은 걸러서 내용어 동사만 추천한다.
    """
    if _collocation_conn is None:
        return []
    rows = _collocation_conn.execute(
        "SELECT verb, freq, sig FROM collocation WHERE noun=? ORDER BY sig DESC LIMIT ?",
        (noun, limit + len(FUNCTION_VERBS))
    ).fetchall()
    out = []
    for r in rows:
        if r['verb'] in FUNCTION_VERBS:
            continue
        out.append({'verb': r['verb'], 'freq': r['freq'], 'sig': round(r['sig'], 1)})
        if len(out) >= limit:
            break
    return out


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
    # XSV/XSA(하다·되다류 파생 접미사)도 서술어다. "편지를 발견했어요"의
    # 발견+하(XSV)를 동사로 안 세면 명사/동사 비율(s2)이 부당하게 치솟아
    # 평범한 하다동사 문장이 경직 문체로 오판된다.
    verbs = [t for t in tokens if t.tag.startswith('V')
             or t.tag.split('-')[0] in ('XSV', 'XSA')]
    char_count = len(sentence.replace(' ', ''))
    total_morph = len(tokens)

    bound_josa_chains = 0
    for i, t in enumerate(tokens):
        if t.tag == 'NNB' and i + 1 < len(tokens) and tokens[i+1].tag.startswith('J'):
            bound_josa_chains += 1

    matched_patterns = []
    matched_groups = []
    lexical_score = 0
    for group_name, weight, patterns in AWKWARD_PATTERN_GROUPS:
        group_hit = False
        for pattern in patterns:
            if re.search(pattern, sentence):
                matched_patterns.append(pattern)
                group_hit = True
        if group_hit:
            matched_groups.append(group_name)
            lexical_score += weight

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

    s4 = min(lexical_score, 40)

    # 시그널 5: 코퍼스 결합 빈도 룩업
    s5 = 0
    awkward_collocations = []
    oov_nouns = []
    if _collocation_conn is not None:
        # 코퍼스에 없는 명사는 s5가 판정 불가로 침묵하는 사각지대다(신조어,
        # 희귀 명사). 어느 명사가 미검수로 통과했는지 소비자(LLM/파이프라인)가
        # 알 수 있도록 별도 필드로 노출한다. 외부 검증(웹 검색 등)의 대상 목록.
        for t in tokens:
            if t.tag.split('-')[0] in ('NNG', 'NNP') and t.form not in oov_nouns:
                if not _is_noun_in_corpus(t.form):
                    oov_nouns.append(t.form)
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
                if freq <= COLLOCATION_WEAK_MAX_FREQ and sig < COLLOCATION_SIG_THRESHOLD:
                    # 결합이 희소하고 유의도도 우연 수준 → 약한 신호
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
        'matched_groups': matched_groups,
        'awkward_collocations': awkward_collocations,
        'oov_nouns': oov_nouns,
    }


def extract_body_sentences(text: str) -> list:
    """마크다운 텍스트에서 평가 대상 문장 리스트 추출.

    제외: 제목, 해시태그 줄, 구분선, 표, 코드블록, 비문장 불릿(명사 나열·항목 표기).
    포함: 본문 평이체 문장, 완결형 문장 불릿.
    """
    lines = text.split('\n')
    body_lines = []
    in_code_block = False

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
            # 블로그 초안은 불릿 안에 완결 문장이 많다. 문장 종결형(마침표 또는
            # 다/요/죠 어미)으로 끝나는 불릿만 평가에 포함하고, 명사 나열이나
            # "가격: 12,000원" 같은 항목 표기는 형태소 통계가 왜곡되므로 제외.
            item = stripped.lstrip('-* ').strip()
            if len(item) >= 6 and re.search(r'(?:[.!?…]|[다요죠])\s*$', item):
                if not re.search(r'[.!?…]$', item):
                    item += '.'
                body_lines.append(item)
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
    all_oov = []
    for r in results:
        for n in r.get('oov_nouns', []):
            if n not in all_oov:
                all_oov.append(n)
    return {
        'file': file_path,
        'total_sentences': len(sentences),
        'suspect_count': len(suspects),
        'average_score': round(avg, 1),
        'threshold': threshold,
        'corpus_signal_active': _collocation_conn is not None,
        'signal_meaning': (
            'corpus_collocation freq=0은 코퍼스 기준 양성(적발)이다. '
            '"거짓양성"은 검사 결과가 아니라 모델이 사후에 붙이는 미검증 라벨일 뿐이다. '
            '해당 명사의 의미를 직접(자기 지식, 필요시 웹검색) 확인하고 그 명사에 자연스러운 동사인지 '
            '같은 부류 명사의 연어로 판정하기 전에는 어떤 적발도 일축하지 마라. '
            '동사가 어긋나면 적절한 동사로 고치고, 자연스러우면 그 치환·의미 비교 근거를 한 줄 남겨라. '
            '"코퍼스가 작아서/명사가 없어서 0" 단독 사유로는 일축 불가다.'
        ),
        'oov_nouns': all_oov,
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
    print(f'  해석 규칙: {result["signal_meaning"]}')
    if result.get('oov_nouns'):
        print(f'  코퍼스 부재 명사(미검수 통과, 외부 검증 권장): {", ".join(result["oov_nouns"][:20])}')
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

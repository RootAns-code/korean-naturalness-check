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
# 관계별 0건 벌점. 단독으로는 임계(17) 미달이 설계 의도다 — 코퍼스 0건의
# 본질적 모호성(어색 vs 자연이지만 뉴스 미수록) 최종 판정은 LLM에 위임하고,
# 소비자는 --threshold 12로 실행해 12~16점 구간을 재판단 밴드로 쓸 수 있다.
# 벌점을 임계 이상으로 올리면 (천천히,오르) 같은 자연-희소 결합이 단독
# 거짓양성이 되는 것을 실측으로 확인함(2026-07-20). 조정 시 eval 실측 필수.
COLLOCATION_SCORE_ZERO = {'nv': 12, 'vn': 12, 'av': 12}
COLLOCATION_SCORE_WEAK = 6          # 약한 유의도 페어당 점수
COLLOCATION_SCORE_MAX = 20          # 시그널 5 최대 점수


def _load_collocation_db():
    if not os.path.exists(_DB_PATH):
        return None, False
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        # 테이블 존재 확인
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='collocation'")
        if cur.fetchone() is None:
            conn.close()
            return None, False
        # rel 컬럼(2.4.0 다중 관계 DB) 여부. 구 DB면 nv만 검사(하위 호환).
        cols = [r[1] for r in conn.execute("PRAGMA table_info(collocation)")]
        return conn, ('rel' in cols)
    except sqlite3.Error:
        return None, False


_collocation_conn, _DB_HAS_REL = _load_collocation_db()


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
    ('직역_명사구', 18, [
        # 동사 관형형이 추상명사를 수식하는 영어 직역 구조. kiwi가 "미는"을
        # 명사+조사로 오분석해 통계(s5)로는 못 잡는 부류라 정규식으로 차단
        # (2026-07-20 실측: "유료 전환을 미는 압박이 됩니다" s5 0점 통과).
        r"[을를]\s*미는\s*(압박|힘|요인|흐름)",
        r"미는\s*압박",
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


def _predicate_stem(tokens, i):
    """i번 토큰이 용언 어간이면 (어간, 소모된 파생 어근 인덱스) 반환. 아니면 (None, None).

    VV/VA뿐 아니라 명사·어근+XSV/XSA 파생 용언(발견하다, 따뜻하다, 우수하다)도
    어간으로 묶는다. 2.3.0까지는 하다류 용언이 통째로 검사 밖이었다(2026-07-19 확인).
    kiwi 0.23+의 규칙/불규칙 하위태그(VV-R 등)는 베이스 태그 비교로 흡수한다.
    """
    t = tokens[i]
    base = t.tag.split('-')[0]
    if base in ('VV', 'VA'):
        return t.form, None
    if base in ('XSV', 'XSA') and i > 0 and tokens[i - 1].tag.split('-')[0] in ('NNG', 'XR'):
        return tokens[i - 1].form + t.form, i - 1
    return None, None


def _extract_collocation_pairs(tokens):
    """문장 토큰열에서 논항·수식 결합 트리플 추출.

    반환: [(rel, left, right), ...] 중복 제거됨.
      rel='nv': 명사 → 후속 용언        예: (편지, 발견하), (기업, 묶이)
      rel='vn': 용언 관형형 → 수식 명사  예: (두껍, 신뢰), (따뜻하, 커피)
      rel='av': 부사 → 후속 용언        예: (급격히, 진행하)
    2.3.0까지는 nv만 검사해 관형어·부사 수식의 어색함("두꺼운 신뢰", "급격히
    진행했다")이 전부 통과했다. 세 관계 모두 같은 함수로 추출하고, 빌드
    스크립트가 이 함수를 그대로 임포트해 DB를 집계하므로 여기를 고치면
    반드시 DB를 재빌드한다(2.1.3 교훈).
    """
    pairs = set()
    noun_hist = []      # [(form, idx)] 최근 명사들 (파생 어근 구분용)
    adv, adv_idx = None, -1
    n = len(tokens)
    for i, t in enumerate(tokens):
        base = t.tag.split('-')[0]
        if base in ('NNG', 'NNP'):
            # "N과 같이/같은", "N보다"의 N은 비교 관용구라 서술 관계가 아님 → 제외
            nxt = tokens[i + 1] if i + 1 < n else None
            nx2 = tokens[i + 2] if i + 2 < n else None
            if (nxt is not None and nxt.tag.startswith('J') and nxt.form in ('과', '와')
                    and nx2 is not None and nx2.form.startswith('같')):
                continue
            if nxt is not None and nxt.tag.startswith('J') and nxt.form == '보다':
                continue
            noun_hist.append((t.form, i))
        elif base == 'MAG':
            adv, adv_idx = t.form, i
        else:
            stem, root_idx = _predicate_stem(tokens, i)
            if stem is None:
                continue
            nxt = tokens[i + 1] if i + 1 < n else None
            # 부사화(V+게): "예쁘게 찍다", "나도 모르게" — 서술어가 아니므로
            # 명사를 소모하지 않고 본용언을 기다린다 (VA뿐 아니라 VV도 해당)
            if nxt is not None and nxt.form == '게' and nxt.tag == 'EC':
                continue
            # 지시 용언은 내용 서술어가 아님 ("노트북이라 그런지" 잡음 방지)
            if t.form in ('그렇', '이렇', '저렇', '어떻', '그러', '이러', '저러'):
                continue
            # vn: 형용사 관형형(ETM)이 바로 다음 명사를 수식.
            # 형용사(VA/XSA)로 제한한다 — 어색 수식("두꺼운 신뢰", "깊은 가격")은
            # 형용사에서 나오고, 일반 동사 관형형("산 신발", "남은 재료")까지 검사하면
            # 뉴스 코퍼스 희소성 때문에 자연 결합이 대량 거짓양성이 된다(2026-07-20 실측).
            is_adjective = t.tag.split('-')[0] in ('VA', 'XSA')
            if (is_adjective and nxt is not None and nxt.tag == 'ETM' and i + 2 < n
                    and tokens[i + 2].tag.split('-')[0] in ('NNG', 'NNP')):
                pairs.add(('vn', stem, tokens[i + 2].form))
            # av: 직전 부사(6형태소 이내)가 이 용언을 수식.
            # '-히' 한자어계 부사로 제한한다 — 보고서투 부사 남용("급격히 진행",
            # "활발히 해결")이 표적이고, 일상 부사(정말/바로/새로/늘)는 자유 결합이라
            # 코퍼스 부재가 어색의 증거가 못 된다(2026-07-20 실측).
            if adv is not None and i - adv_idx <= 6:
                if adv.endswith('히') and adv != '같이':
                    pairs.add(('av', adv, stem))
                adv = None
            # nv: 가장 가까운 선행 명사(10형태소 이내). 하다류 파생 어근 자신은 제외
            # ("편지를 발견하다"에서 (발견, 발견하)가 아니라 (편지, 발견하))
            for form, idx in reversed(noun_hist):
                if root_idx is not None and idx == root_idx:
                    continue
                if i - idx <= 10:
                    pairs.add(('nv', form, stem))
                break
            noun_hist.clear()
    return list(pairs)


def _extract_noun_verb_pairs(tokens):
    """하위 호환: nv 결합만 (noun, verb) 페어로 반환."""
    return [(l, r) for rel, l, r in _extract_collocation_pairs(tokens) if rel == 'nv']


def _lookup_collocation(rel, left, right):
    """코퍼스 DB에서 (rel, left, right) 결합 빈도와 유의도 조회.

    반환: (freq, sig) 또는 None (결합이 DB에 없음).
    DB 컬럼명은 하위 호환을 위해 noun(=left)/verb(=right)를 유지한다.
    """
    if _collocation_conn is None:
        return None
    if _DB_HAS_REL:
        row = _collocation_conn.execute(
            "SELECT freq, sig FROM collocation WHERE rel=? AND noun=? AND verb=?",
            (rel, left, right)).fetchone()
    else:
        if rel != 'nv':
            return None
        row = _collocation_conn.execute(
            "SELECT freq, sig FROM collocation WHERE noun=? AND verb=?",
            (left, right)).fetchone()
    if row is None:
        return None
    return (row['freq'], row['sig'])


# 연관어 목록에서 제외할 기능성 용언. 어떤 명사와도 결합해 sig 상위를
# 도배하지만("기업에 대하", "~를 위하") 적발 근거 자료로는 정보 가치가 없다.
FUNCTION_VERBS = frozenset([
    '있', '없', '하', '되', '대하', '위하', '통하', '따르', '관하', '의하',
    '인하', '비롯하', '같', '이러하', '그러하', '아니하', '못하', '말미암',
])


def _corpus_associations(rel, left, right, limit=5):
    """적발된 결합의 축 단어가 코퍼스에서 자주 만난 연관어 상위 N개 (sig 기준).

    문장을 보지 않는 단어 단위 빈도 통계라 문맥 비인지다. 대체 표현 후보가
    아니라 "이 단어는 원래 이런 결합으로 쓰인다"는 적발 근거 자료이며,
    수정 표현은 소비자(LLM)가 원문 의미를 기준으로 새로 짠다.

    nv: 그 명사(left)와 자주 결합하는 용언들.
    vn: 그 명사(right)를 자주 수식하는 관형 용언들 (예: 신뢰 → 두텁, 깊).
    av: 그 용언(right)과 자주 어울리는 부사들.
    기능성 용언(FUNCTION_VERBS)은 걸러서 내용어만 남긴다.
    """
    if _collocation_conn is None:
        return []
    if _DB_HAS_REL:
        if rel == 'nv':
            q, key = ("SELECT verb AS alt, freq, sig FROM collocation "
                      "WHERE rel='nv' AND noun=? ORDER BY sig DESC LIMIT ?"), left
        elif rel == 'vn':
            q, key = ("SELECT noun AS alt, freq, sig FROM collocation "
                      "WHERE rel='vn' AND verb=? ORDER BY sig DESC LIMIT ?"), right
        else:
            q, key = ("SELECT noun AS alt, freq, sig FROM collocation "
                      "WHERE rel='av' AND verb=? ORDER BY sig DESC LIMIT ?"), right
    else:
        if rel != 'nv':
            return []
        q, key = ("SELECT verb AS alt, freq, sig FROM collocation "
                  "WHERE noun=? ORDER BY sig DESC LIMIT ?"), left
    rows = _collocation_conn.execute(q, (key, limit + len(FUNCTION_VERBS))).fetchall()
    out = []
    for r in rows:
        if r['alt'] in FUNCTION_VERBS:
            continue
        out.append({'verb': r['alt'], 'freq': r['freq'], 'sig': round(r['sig'], 1)})
        if len(out) >= limit:
            break
    return out


def _is_left_in_corpus(rel, left):
    """결합의 왼쪽 요소가 해당 관계로 코퍼스에 한 번이라도 등장하는지 확인.

    코퍼스에 없는 요소는 빌드 누락이거나 매우 희귀할 가능성이 커서
    결합을 판정 못 함. 이 경우 그 결합에 대해 시그널 5는 침묵.
    """
    if _collocation_conn is None:
        return False
    if _DB_HAS_REL:
        row = _collocation_conn.execute(
            "SELECT 1 FROM collocation WHERE rel=? AND noun=? LIMIT 1",
            (rel, left)).fetchone()
    else:
        if rel != 'nv':
            return False
        row = _collocation_conn.execute(
            "SELECT 1 FROM collocation WHERE noun=? LIMIT 1", (left,)).fetchone()
    return row is not None


def _is_noun_in_corpus(noun):
    """하위 호환: 명사가 nv 관계로 코퍼스에 등장하는지 확인 (oov_nouns 판정용)."""
    return _is_left_in_corpus('nv', noun)


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
        triples = _extract_collocation_pairs(tokens)
        for rel, left, right in triples:
            # 왼쪽 요소가 코퍼스에 아예 없으면 판정 불가 (빌드 누락). 침묵.
            if not _is_left_in_corpus(rel, left):
                continue
            result = _lookup_collocation(rel, left, right)
            if result is None:
                # 왼쪽 요소는 코퍼스에 있는데 이 결합은 0건 → 강한 신호
                s5 += COLLOCATION_SCORE_ZERO[rel]
                awkward_collocations.append({
                    'rel': rel,
                    'noun': left,
                    'verb': right,
                    'freq': 0,
                    'sig': 0.0,
                    'corpus_associations': _corpus_associations(rel, left, right, limit=5),
                })
            else:
                freq, sig = result
                if freq <= COLLOCATION_WEAK_MAX_FREQ and sig < COLLOCATION_SIG_THRESHOLD:
                    # 결합이 희소하고 유의도도 우연 수준 → 약한 신호
                    s5 += COLLOCATION_SCORE_WEAK
                    awkward_collocations.append({
                        'rel': rel,
                        'noun': left,
                        'verb': right,
                        'freq': freq,
                        'sig': round(sig, 1),
                        'corpus_associations': _corpus_associations(rel, left, right, limit=5),
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

    # 줄 단위로 문장을 나눈다. 줄 안에 종결부호가 있으면 추가 분할하고,
    # 종결부호가 없는 줄은 그 줄 전체를 한 문장으로 본다 — SNS 톤(마침표 생략,
    # 줄바꿈이 문장부호를 대신)의 텍스트에서 여러 줄이 한 덩어리로 합쳐져
    # 형태소 통계가 왜곡되던 문제 방지.
    sentences = []
    for line in body_lines:
        for piece in re.split(r'(?<=[.!?…])\s+', line):
            piece = piece.strip()
            if piece and len(piece) >= 6:
                sentences.append(piece)
    return sentences


def evaluate_file(file_path: str, threshold: int = 17) -> dict:
    """파일을 평가해 결과 dict 반환."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = extract_body_sentences(text)
    results = [naturalness_score(s) for s in sentences]
    suspects = [r for r in results if r['score'] >= threshold]
    suspects.sort(key=lambda r: -r['score'])
    # 12점(단일 강신호)부터 임계 미만까지는 "통과"가 아니라 "미보고"다.
    # 소비자 모델이 이 구간을 놓치지 않도록 별도 필드로 항상 내보낸다.
    review_band = [r for r in results if 12 <= r['score'] < threshold]
    review_band.sort(key=lambda r: -r['score'])
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
            '"코퍼스가 작아서/명사가 없어서 0" 단독 사유로는 일축 불가다. '
            'corpus_associations는 문맥을 보지 않은 단어 빈도 연관어다. 대체 표현 후보로 쓰지 말고 '
            '적발 근거 확인용으로만 참고하며, 수정 문장은 문제 단어의 치환이 아니라 '
            '원문이 전달하려는 의미 관계(누가 무엇을 하는지)를 기준으로 새로 작성하라.'
        ),
        'oov_nouns': all_oov,
        'suspects': suspects,
        'review_band_count': len(review_band),
        'review_band': review_band,
        'review_band_meaning': (
            'review_band는 12점 이상 임계 미만이라 suspects에서 빠진 문장이다. '
            '임계 아래는 검사 통과가 아니라 미보고일 뿐이다. 이 결과를 소비하는 모델은 '
            'suspects 처리 후 review_band를 반드시 훑는다. 한 글자 명사, 고유명사의 부분 문자열, '
            'oov 명사 같은 파서 조각 적발을 제외하고 온전한 결합이나 어휘 패턴이 남는 문장만 '
            '실제로 읽고 판정한다. suspects처럼 재작성을 강제하는 구간이 아니라 어색 여부를 '
            '판정하는 구간이며 어색하면 고치고 자연스러우면 그대로 둔다. '
            'oov_nouns가 낀 문장은 코퍼스 미검수 상태이므로 결합이 자연스러운지 자기 지식으로 '
            '한 번 더 확인한다. suspects만 처리하고 끝내면 이 검사를 절반만 쓴 것이다.'
        ),
        'all_results': results,
    }


def print_text_report(result: dict, top: int = 10):
    """기본 텍스트 출력."""
    print('한국어 자연스러움 점검:')
    print(f'  검사 문장 수: {result["total_sentences"]}')
    print(f'  의심 문장 수: {result["suspect_count"]} (임계 점수 {result["threshold"]} 이상)')
    if result.get('review_band_count'):
        print(f'  재판단 밴드: {result["review_band_count"]}문장 (12점 이상 임계 미만 — 미보고일 뿐 통과가 아니므로 suspects 처리 후 반드시 판정)')
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
            rel_label = {'nv': '명사-용언', 'vn': '관형-명사', 'av': '부사-용언'}
            for c in r['awkward_collocations'][:3]:
                rel = rel_label.get(c.get('rel', 'nv'), c.get('rel'))
                print(f'      어색 결합[{rel}]: ({c["noun"]}, {c["verb"]}) freq={c["freq"]} sig={c["sig"]}')
        print()

    print(f'전체 결과: WARN ({result["suspect_count"]} / {result["total_sentences"]} 문장 의심)')
    print('  지침: 의심 문장을 자연스러운 한국어로 다듬은 뒤 다시 검사할 것.')
    print('  지침: 어색 결합은 단어만 바꾸기보다 의미 관계(누가 무엇을 하는지)를 다시 세우는 수정을 우선 검토할 것.')


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

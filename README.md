# korean-naturalness-check

한국어 텍스트의 자연스러움을 정량 평가하는 Claude 스킬입니다. 번역투, 매거진투, AI투, 직역체 같은 어색한 표현을 형태소 분석과 코퍼스 통계로 검출합니다.

## 왜 필요한가

LLM은 자기가 쓴 한국어를 자기가 검수하면 자연스럽다고 판정하는 편향이 있습니다. 이 스킬은 LLM 자체 검수와 다른 분포의 **외부 통계 시그널**을 제공해, 자체 검수가 놓치는 어색함을 잡습니다.

## 동작 방식

`scripts/check_naturalness.py`가 문장별로 5개 시그널을 합산해 0~100 점수를 매기고, 임계(17점) 이상을 의심 문장으로 보고합니다.

| 시그널 | 내용 |
|---|---|
| 1. 명사 밀도 | 명사 나열 위주의 번역투 문장 검출 |
| 2. 명사/동사 비율 | 동사가 빈약한 경직 문체 검출 |
| 3. 의존명사+조사 연쇄 | "~것으로의" 류 직역체 검출 |
| 4. 어색 어휘 사전 | 현상 그룹 단위 정규식 사전(`AWKWARD_PATTERN_GROUPS`) 매칭, 확실(18)/강(12)/약(6) 3단계 가중 |
| 5. 코퍼스 결합 빈도 | 동봉 DB에서 (명사, 동사) 결합 빈도 룩업 — 사전에 없는 어색 결합도 자동 검출 |

시그널 5의 참조 데이터(`assets/kor_collocation.db`, 약 12.5MB)는 Leipzig kor_news_2022_1M 코퍼스의 원문 100만 문장을 kiwipiepy로 형태소 분석해 빌드한 명사+동사 결합 빈도 SQLite DB입니다(163,272 페어). 런타임 검사기와 같은 추출기로 집계해 DB-런타임 분포가 일치합니다. 빌드 절차는 `scripts/build_collocation_data.py` 참고 — 사용자는 실행할 필요 없습니다.

코퍼스에 없는 명사(2022 이후 신조어 등)는 판정 불가로 침묵하는 대신 `oov_nouns` 필드로 노출되어, 소비자(LLM/파이프라인)가 웹 검색 등 외부 검증으로 이어갈 수 있습니다.

## 설치

```bash
git clone https://github.com/RootAns-code/korean-naturalness-check.git ~/.claude/skills/korean-naturalness-check
```

Claude.ai 웹에서는 저장소를 zip으로 받아 스킬 업로드하면 됩니다. 의존성은 `kiwipiepy` 하나이며 스킬이 첫 실행 시 설치를 안내합니다.

## 사용

Claude Code 또는 Claude.ai에서 한국어 본문을 주고 다음과 같이 요청하면 스킬이 트리거됩니다.

> 이 글 번역투 점검해줘 / 자연스러움 검수해줘 / 이 본문 평가해줘

직접 실행할 수도 있습니다.

```bash
python3 scripts/check_naturalness.py <마크다운 파일> --json
```

## 검증 정확도

`eval/run_eval.py` 라벨셋(자연 70·어색 72문장, 2.3.0 기준) 실측:

- 재현율 86.1%, 거짓양성률 10.0%, 정밀도 89.9%
- 남은 거짓양성은 대부분 뉴스 코퍼스에 없는 일상 결합의 s5 적발 — 보고 단계에서 LLM이 의미 검증 후 걸러냄
- 시그널·임계값·사전을 수정할 때는 `python3 eval/run_eval.py`로 전후 수치를 비교 (회귀 방지)

## 버전

현재 버전과 변경 이력은 [SKILL.md](SKILL.md) 프론트매터와 [CHANGELOG.md](CHANGELOG.md)를 참고하세요. [Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다.

## 라이선스 표기

동봉 코퍼스 DB의 원천 데이터: [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Korean) — kor_news_2022_1M, CC BY 4.0

#!/usr/bin/env python3
"""라벨셋 기반 검수 정확도 측정 하네스.

자연/어색 라벨 문장셋에 대해 check_naturalness의 문장 점수를 계산하고
정밀도·재현율·거짓양성률을 보고한다. 임계값·시그널·사전·DB를 바꾸는 모든
변경은 이 하네스로 전후 수치를 비교해 회귀 여부를 판정한다
(2.1.1 sig 임계값 회귀에서 얻은 원칙: 보정은 추론이 아니라 실측으로).

사용법:
  python3 eval/run_eval.py            # 요약 보고
  python3 eval/run_eval.py -v         # 거짓양성/거짓음성 문장 목록 포함
  python3 eval/run_eval.py --threshold 17
"""
import argparse
import os
import sys
from collections import Counter

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_EVAL_DIR, '..', 'scripts'))

from check_naturalness import naturalness_score, extract_body_sentences  # noqa: E402

NATURAL_PATH = os.path.join(_EVAL_DIR, 'natural_sentences.txt')
AWKWARD_PATH = os.path.join(_EVAL_DIR, 'awkward_sentences.txt')


def load_sentences(path):
    sentences = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sentences.append(line)
    return sentences


def signal_contributors(breakdown):
    """0점 초과 시그널 키 목록."""
    return [k for k, v in breakdown.items() if v > 0]


def run_classification(threshold, verbose):
    natural = load_sentences(NATURAL_PATH)
    awkward = load_sentences(AWKWARD_PATH)

    nat_results = [naturalness_score(s) for s in natural]
    awk_results = [naturalness_score(s) for s in awkward]

    fp = [r for r in nat_results if r['score'] >= threshold]
    tn = len(nat_results) - len(fp)
    tp = [r for r in awk_results if r['score'] >= threshold]
    fn = [r for r in awk_results if r['score'] < threshold]

    nat_mean = sum(r['score'] for r in nat_results) / max(len(nat_results), 1)
    awk_mean = sum(r['score'] for r in awk_results) / max(len(awk_results), 1)
    precision = len(tp) / max(len(tp) + len(fp), 1)
    recall = len(tp) / max(len(awk_results), 1)
    fp_rate = len(fp) / max(len(nat_results), 1)

    print('=' * 62)
    print(f'분류 성능 (임계 {threshold}점)')
    print('=' * 62)
    print(f'자연 문장 {len(nat_results)}개: 평균 {nat_mean:.1f}점, '
          f'거짓양성 {len(fp)}개 ({fp_rate:.1%}), 정상 통과 {tn}개')
    print(f'어색 문장 {len(awk_results)}개: 평균 {awk_mean:.1f}점, '
          f'검출 {len(tp)}개 (재현율 {recall:.1%}), 놓침 {len(fn)}개')
    print(f'정밀도 {precision:.1%}  |  평균 점수 격차 {awk_mean - nat_mean:+.1f}점')

    fp_signals = Counter()
    for r in fp:
        for k in signal_contributors(r['breakdown']):
            fp_signals[k] += 1
    if fp_signals:
        print(f'거짓양성 기여 시그널: '
              + ', '.join(f'{k}×{v}' for k, v in fp_signals.most_common()))

    # 자연 문장에서 s5 거짓 벌점을 받은 건수 (임계 미달이라도 벌점 자체가 잡음)
    nat_s5_hits = [r for r in nat_results if r['breakdown']['corpus_collocation'] > 0]
    print(f'자연 문장 중 s5(코퍼스) 벌점 발생: {len(nat_s5_hits)}개 '
          f'({len(nat_s5_hits) / max(len(nat_results), 1):.1%}) — 0에 가까울수록 좋음')

    if verbose:
        if fp:
            print('\n--- 거짓양성 (자연인데 의심 판정) ---')
            for r in sorted(fp, key=lambda x: -x['score']):
                print(f'  [{r["score"]:5.1f}] {r["sentence"]}')
                print(f'          matched={r["matched"]} '
                      f'colloc={[(c["noun"], c["verb"]) for c in r["awkward_collocations"]]}')
        if fn:
            print('\n--- 거짓음성 (어색한데 통과) ---')
            for r in sorted(fn, key=lambda x: -x['score']):
                print(f'  [{r["score"]:5.1f}] {r["sentence"]}')
        if nat_s5_hits:
            print('\n--- 자연 문장 s5 벌점 상세 ---')
            for r in nat_s5_hits:
                pairs = [(c['noun'], c['verb']) for c in r['awkward_collocations']]
                print(f'  [{r["score"]:5.1f}] {pairs} ← {r["sentence"]}')

    return {'precision': precision, 'recall': recall, 'fp_rate': fp_rate}


STRUCTURE_MD = """# 테스트 제목

첫 문단 문장은 반드시 평가에 포함되어야 합니다.

- 불릿이지만 완결된 문장은 평가 대상에 들어가야 합니다.
- 명사 나열 불릿
- 가격: 12,000원

두 번째 문단 문장도 평가에 포함됩니다.
"""


def run_structure_checks():
    print()
    print('=' * 62)
    print('문서 파싱 구조 점검 (extract_body_sentences)')
    print('=' * 62)
    sents = extract_body_sentences(STRUCTURE_MD)
    checks = [
        ('첫 본문 문단 포함', any('첫 문단' in s for s in sents)),
        ('완결형 불릿 문장 포함', any('불릿이지만' in s for s in sents)),
        ('명사 나열 불릿 제외', not any('명사 나열' in s for s in sents)),
        ('비문장 불릿(가격표) 제외', not any('12,000' in s for s in sents)),
        ('일반 문단 포함', any('두 번째 문단' in s for s in sents)),
    ]
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    return all(ok for _, ok in checks)


def main():
    parser = argparse.ArgumentParser(description='라벨셋 기반 정확도 측정')
    parser.add_argument('--threshold', type=int, default=17)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    run_classification(args.threshold, args.verbose)
    run_structure_checks()


if __name__ == '__main__':
    main()

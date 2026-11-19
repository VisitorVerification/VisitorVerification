"""판정 기준값 실험: 그래프와 결과 표 (A 담당, 인수인계서 5.4).

experiments/distances.csv를 읽어
- 조건별 본인 거리 vs 타인 거리 히스토그램을 experiments/plots/에 PNG로 저장하고
- 보고서용 결과 정리 표를 콘솔에 출력하고 experiments/summary.csv로 저장한다.

실행 (프로젝트 루트에서):
    python experiments/plot_distances.py                  # 기준값 후보(타인 최솟값의 90%)로 표 계산
    python experiments/plot_distances.py --threshold 0.45 # 이 기준값으로 표 계산

"타인 거리" = 등록자 역할의 min_other_distance + 미등록자 역할(unknown)의 nearest_distance
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 프로젝트 루트

import matplotlib
matplotlib.use("Agg")                     # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np

from config import MATCH_THRESHOLD

CONDITIONS = ["front_bright", "side_bright", "front_dark", "front_dark_clahe", "far"]
UNKNOWN_WHO = "unknown"
CSV_PATH = os.path.join("experiments", "distances.csv")
PLOTS_DIR = os.path.join("experiments", "plots")
SUMMARY_PATH = os.path.join("experiments", "summary.csv")
MARGIN = 0.9                              # 기준값 후보 = 타인 거리 최솟값 × 0.9 (10% 아래)

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 깨짐 방지 (Windows 맑은 고딕)
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────
# 데이터 읽기
# ──────────────────────────────────────────────
def _float(v: str) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load(path: str = CSV_PATH) -> dict[str, dict[str, list[float]]]:
    """{조건: {"self": [...], "other": [...]}}"""
    data = {c: {"self": [], "other": []} for c in CONDITIONS}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            cond = r.get("condition", "")
            if cond not in data:
                continue
            if r.get("who") == UNKNOWN_WHO:
                v = _float(r.get("nearest_distance"))
                if v is not None:
                    data[cond]["other"].append(v)
            else:
                s, o = _float(r.get("self_distance")), _float(r.get("min_other_distance"))
                if s is not None:
                    data[cond]["self"].append(s)
                if o is not None:
                    data[cond]["other"].append(o)
    return data


# ──────────────────────────────────────────────
# 그래프
# ──────────────────────────────────────────────
def _hist(ax, self_d, other_d, title, candidate):
    values = self_d + other_d
    bins = np.linspace(0, max(1.0, max(values) if values else 1.0), 41)
    if self_d:
        ax.hist(self_d, bins=bins, alpha=0.6, color="#3B82F6", label=f"본인 ({len(self_d)})")
    if other_d:
        ax.hist(other_d, bins=bins, alpha=0.6, color="#EF4444", label=f"타인 ({len(other_d)})")
    ax.axvline(MATCH_THRESHOLD, color="black", linestyle="--", linewidth=1,
               label=f"현재 기준값 {MATCH_THRESHOLD:.3f}")
    if candidate is not None:
        ax.axvline(candidate, color="#16A34A", linestyle="-", linewidth=1.5,
                   label=f"기준값 후보 {candidate:.3f}")
    ax.set_title(title)
    ax.set_xlabel("거리 (1 − 코사인 유사도, 작을수록 비슷)")
    ax.set_ylabel("프레임 수")
    ax.legend(fontsize=8)


def save_plots(data, candidate) -> list[str]:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    saved = []
    for cond in CONDITIONS:
        d = data[cond]
        if not d["self"] and not d["other"]:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        _hist(ax, d["self"], d["other"], cond, candidate)
        path = os.path.join(PLOTS_DIR, f"{cond}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    all_self = [v for c in CONDITIONS for v in data[c]["self"]]
    all_other = [v for c in CONDITIONS for v in data[c]["other"]]
    if all_self or all_other:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _hist(ax, all_self, all_other, "전체 조건", candidate)
        path = os.path.join(PLOTS_DIR, "all.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)
    return saved


# ──────────────────────────────────────────────
# 결과 표
# ──────────────────────────────────────────────
def summarize(data, threshold) -> list[dict]:
    rows = []
    for cond in CONDITIONS:
        s, o = data[cond]["self"], data[cond]["other"]
        rows.append({
            "조건": cond,
            "본인 거리 평균": f"{np.mean(s):.4f}" if s else "-",
            "타인 거리 최솟값": f"{min(o):.4f}" if o else "-",
            "기준값에서 본인 인식": f"{sum(v <= threshold for v in s)}/{len(s)} ({100 * np.mean([v <= threshold for v in s]):.0f}%)" if s else "-",
            "기준값에서 오인식": f"{sum(v <= threshold for v in o)}/{len(o)}" if o else "-",
        })
    return rows


def print_table(rows, threshold) -> None:
    cols = list(rows[0].keys())
    print(f"\n📋 결과 정리 표 (기준값 {threshold:.4f})\n")
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def save_summary(rows) -> None:
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="기준값 실험 결과 그래프와 표")
    parser.add_argument("--threshold", type=float, help="표를 계산할 기준값 (없으면 기준값 후보 사용)")
    args = parser.parse_args(argv)

    if not os.path.isfile(CSV_PATH):
        print(f"{CSV_PATH}가 없습니다. 먼저 experiments/collect_distances.py로 측정해 주세요.")
        return 1
    data = load()

    all_other = [v for c in CONDITIONS for v in data[c]["other"]]
    min_other = min(all_other) if all_other else None
    candidate = min_other * MARGIN if min_other is not None else None

    print(f"📂 {CSV_PATH}")
    for cond in CONDITIONS:
        print(f"  {cond:<18} 본인 {len(data[cond]['self']):>3}개   타인 {len(data[cond]['other']):>3}개")
    if min_other is not None:
        print(f"\n🔎 타인 거리 최솟값(전체): {min_other:.4f}")
        print(f"🎯 기준값 후보 (× {MARGIN}): {candidate:.4f}   (현재 config MATCH_THRESHOLD = {MATCH_THRESHOLD})")
    else:
        print("\n⚠️ 타인 거리 데이터가 없어 기준값 후보를 계산할 수 없습니다 (미등록자 역할 측정이 필요해요).")

    threshold = args.threshold if args.threshold is not None else (candidate if candidate is not None else MATCH_THRESHOLD)
    rows = summarize(data, threshold)
    print_table(rows, threshold)
    save_summary(rows)

    # CLAHE 비교 (5.4)
    dark, clahe = data["front_dark"]["self"], data["front_dark_clahe"]["self"]
    if dark and clahe:
        better = np.mean(clahe) < np.mean(dark)
        print(f"\n🌗 CLAHE 비교 (본인 거리 평균): front_dark {np.mean(dark):.4f} → front_dark_clahe {np.mean(clahe):.4f}"
              f"  → {'좋아짐 (USE_CLAHE = True 검토, 바꾸면 전원 재등록)' if better else '좋아지지 않음 (USE_CLAHE = False 유지)'}")

    saved = save_plots(data, candidate)
    print(f"\n✅ 표 저장: {SUMMARY_PATH}")
    for p in saved:
        print(f"✅ 그래프 저장: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

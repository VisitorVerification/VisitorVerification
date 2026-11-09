"""CSV를 읽어 본인 거리와 타인 거리 히스토그램을 그리고 PNG로 저장한다 (5.4).

python experiments/plot_distances.py
python experiments/plot_distances.py --csv experiments/distances.csv --threshold 0.45

결과:
  experiments/plots/hist_all.png, hist_<조건>.png   히스토그램
  experiments/plots/summary.csv                      보고서용 결과 정리 표
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

CONDITIONS = ["front_bright", "side_bright", "front_dark", "front_dark_clahe", "far"]
SELF_COLOR, OTHER_COLOR = "#2a7de1", "#e0553d"


def _float(s):
    try:
        return float(s) if s not in ("", None) else None
    except ValueError:
        return None


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["self_distance"] = _float(r.get("self_distance"))
        r["min_other_distance"] = _float(r.get("min_other_distance"))
    return rows


def split(rows):
    self_d = np.array([r["self_distance"] for r in rows if r["self_distance"] is not None])
    other_d = np.array([r["min_other_distance"] for r in rows if r["min_other_distance"] is not None])
    return self_d, other_d


def plot(self_d, other_d, threshold, title, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1.2, 49)
    if self_d.size:
        ax.hist(self_d, bins=bins, alpha=0.75, color=SELF_COLOR, label=f"self (n={self_d.size})")
    if other_d.size:
        ax.hist(other_d, bins=bins, alpha=0.75, color=OTHER_COLOR, label=f"other (n={other_d.size})")
        ax.axvline(other_d.min(), color=OTHER_COLOR, ls=":", lw=1.2, label=f"min other = {other_d.min():.3f}")
    ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("distance (1 - cosine similarity)")
    ax.set_ylabel("frames")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("experiments", "distances.csv"))
    ap.add_argument("--out", default=os.path.join("experiments", "plots"))
    ap.add_argument("--threshold", type=float, default=config.MATCH_THRESHOLD,
                    help="표에 쓸 기준값 (기본: config.MATCH_THRESHOLD)")
    ap.add_argument("--margin", type=float, default=0.10, help="타인 최솟값보다 몇 %% 아래를 후보로 할지")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV가 없습니다: {args.csv}  (먼저 collect_distances.py 실행)")
        return 1
    rows = load(args.csv)
    os.makedirs(args.out, exist_ok=True)
    th = args.threshold

    self_all, other_all = split(rows)
    plot(self_all, other_all, th, "all conditions", os.path.join(args.out, "hist_all.png"))

    summary = []
    for cond in CONDITIONS + ["all"]:
        sub = rows if cond == "all" else [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        s, o = split(sub)
        if cond != "all":
            plot(s, o, th, cond, os.path.join(args.out, f"hist_{cond}.png"))
        summary.append({
            "condition": cond,
            "self_mean": f"{s.mean():.4f}" if s.size else "",
            "other_min": f"{o.min():.4f}" if o.size else "",
            "self_recognized": f"{(s <= th).sum()}/{s.size} ({(s <= th).mean() * 100:.0f}%)" if s.size else "",
            "false_accept": f"{(o <= th).sum()}/{o.size}" if o.size else "",
        })

    sum_path = os.path.join(args.out, "summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["조건", "본인 거리 평균", "타인 거리 최솟값", "기준값에서 본인 인식", "기준값에서 오인식"])
        for r in summary:
            w.writerow([r["condition"], r["self_mean"], r["other_min"], r["self_recognized"], r["false_accept"]])

    print(f"기준값 {th:.3f} 기준 결과 (그림·표: {args.out})")
    print(f"{'조건':<18}{'본인 평균':>10}{'타인 최소':>10}{'본인 인식':>16}{'오인식':>10}")
    for r in summary:
        print(f"{r['condition']:<20}{r['self_mean']:>10}{r['other_min']:>12}{r['self_recognized']:>16}{r['false_accept']:>10}")

    if other_all.size:
        cand = other_all.min() * (1 - args.margin)
        rate = (self_all <= cand).mean() * 100 if self_all.size else 0
        print(f"\n타인 거리 최솟값 {other_all.min():.4f} → 기준값 후보 {cand:.4f} "
              f"({args.margin * 100:.0f}% 아래). 이 값에서 본인 인식 {rate:.0f}%")
        print("본인 인식이 부족하면 등록 사진을 보강하고 다시 측정하세요. 확정값은 config.MATCH_THRESHOLD에.")

    # CLAHE 비교
    dark = [r["self_distance"] for r in rows if r["condition"] == "front_dark" and r["self_distance"] is not None]
    dark_c = [r["self_distance"] for r in rows if r["condition"] == "front_dark_clahe" and r["self_distance"] is not None]
    if dark and dark_c:
        a, b = np.mean(dark), np.mean(dark_c)
        verdict = "CLAHE가 본인 거리를 줄였습니다 → USE_CLAHE = True 검토 (전원 재등록 필요)" if b < a \
            else "CLAHE 효과 없음 → USE_CLAHE = False 유지"
        print(f"\nCLAHE 비교: front_dark 본인 평균 {a:.4f}, front_dark_clahe {b:.4f}\n  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

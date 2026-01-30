from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires matplotlib. Install it (e.g. `pip install matplotlib`).") from e


PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


RUNS: List[Tuple[str, Path, str]] = [
    ("small_one_kit", Path("artifacts/eval/small_one_kit"), "Small / One-kit"),
    ("small_all_kits", Path("artifacts/eval/small_all_kits"), "Small / All-kits"),
    ("big_one_kit", Path("artifacts/eval/big_one_kit"), "Big / One-kit"),
    ("big_all_kits", Path("artifacts/eval/big_all_kits"), "Big / All-kits"),
]

# Optional per-kit FAD CSVs (only where you ran `--fadtk-per-kit`).
PER_KIT_FAD: Dict[str, Path] = {
    "small_all_kits": Path("artifacts/eval/fadtk_per_kit_small_all_kits/fadtk_per_kit.csv"),
}


def _paper_model_label_from_stem(stem: str) -> str:
    s = str(stem)
    for prefix in ("expressivegrid_to_", "expressivegrid-to-", "eg_", "eg-"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s.replace("__", "_").strip("_")


def _fit_label(s: str, *, max_chars: int = 22, wrap_at: int = 12) -> str:
    s = str(s)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    if len(s) > wrap_at:
        import textwrap

        return "\n".join(textwrap.wrap(s, width=wrap_at))
    return s


def load_eval(out_dir: Path) -> Tuple[dict, pd.DataFrame]:
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    items = pd.read_csv(out_dir / "items.csv")
    return summary, items


def base_systems(summary: dict) -> List[str]:
    # preserve eval ordering (no sort)
    return [k for k in summary["systems"].keys() if not k.endswith(("_oracle", "_random"))]


def ckpt_from_summary(summary: dict, system: str) -> Optional[Path]:
    p = summary["systems"].get(system, {}).get("ckpt")
    return Path(p) if isinstance(p, str) and p else None


def label_for_system(summary: dict, system: str) -> str:
    ckpt = ckpt_from_summary(summary, system)
    return _paper_model_label_from_stem(ckpt.stem) if ckpt is not None else str(system)


def train_metrics_csv_for_system(summary: dict, system: str) -> Optional[Path]:
    ckpt = ckpt_from_summary(summary, system)
    if ckpt is None:
        return None
    p = Path(str(ckpt) + ".metrics.csv")
    return p if p.is_file() else None


def set_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 9,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.3,
        }
    )


def _mean_std(d: Any) -> Tuple[float, float]:
    if isinstance(d, dict) and isinstance(d.get("mean"), (int, float)) and math.isfinite(float(d["mean"])):
        mu = float(d["mean"])
        sd = float(d.get("std")) if isinstance(d.get("std"), (int, float)) and math.isfinite(float(d.get("std"))) else float("nan")
        return mu, sd
    if isinstance(d, (int, float)) and math.isfinite(float(d)):
        return float(d), float("nan")
    return float("nan"), float("nan")


def _fmt(mu: float, sd: float, digits: int) -> str:
    if not math.isfinite(mu):
        return ""
    if math.isfinite(sd):
        return f"{mu:.{digits}f}±{sd:.{digits}f}"
    return f"{mu:.{digits}f}"


def _fad_key(sys_dict: dict) -> Optional[str]:
    for k in sys_dict.keys():
        if str(k).startswith("fad_fadtk_"):
            return str(k)
    return None


def get_fad_cell(summary: dict, system: str) -> Tuple[str, str]:
    """
    Returns (label, value_str) where label is 'FAD∞' if available else 'FAD'.
    """
    sysd = summary.get("systems", {}).get(system, {}) or {}
    k = _fad_key(sysd)
    if not k:
        return ("", "")
    block = sysd.get(k, {}) or {}
    fi = (block.get("fad_inf", {}) or {})
    v_inf = fi.get("fad_inf", None)
    if isinstance(v_inf, (int, float)) and math.isfinite(float(v_inf)):
        return ("FAD∞", f"{float(v_inf):.3f}")
    v = block.get("fad", None)
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return ("FAD", f"{float(v):.3f}")
    return ("", "")


def plot_training_curves(axs: List[Any], summary: dict, systems: List[str]) -> None:
    for ax, sys in zip(axs, systems):
        label = _fit_label(label_for_system(summary, sys), max_chars=26, wrap_at=14)
        p = train_metrics_csv_for_system(summary, sys)
        if p is None:
            ax.set_title(f"{label}\n(no train log)")
            ax.set_axis_off()
            continue
        df = pd.read_csv(p).sort_values("step")
        if not {"step", "train_loss", "val_loss"}.issubset(df.columns):
            ax.set_title(f"{label}\n(bad train log)")
            ax.set_axis_off()
            continue
        x = df["step"].to_numpy()
        tr = df["train_loss"].to_numpy()
        va = df["val_loss"].to_numpy()
        ax.plot(x, tr, color="C0", label="train")
        ax.plot(x, va, color="C1", label="val")
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("NLL" if ax is axs[0] else "")


def plot_token_acc_heatmap(ax: Any, items: pd.DataFrame, systems: List[str], summary: dict) -> None:
    cb_cols = [c for c in items.columns if c.startswith("token_acc_cb") and items[c].notna().any()]
    cb_cols = sorted(cb_cols, key=lambda s: int(s.replace("token_acc_cb", ""))) if cb_cols else []
    if not cb_cols:
        ax.set_title("Token acc per codebook (no cb cols)")
        ax.set_axis_off()
        return
    per_cb = items.groupby("system")[cb_cols].mean().reindex(systems)
    arr = per_cb.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Token acc per codebook")
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([_fit_label(label_for_system(summary, s), max_chars=18, wrap_at=10) for s in systems])
    ax.set_xticks(range(len(cb_cols)))
    ax.set_xticklabels([c.replace("token_acc_cb", "cb") for c in cb_cols], rotation=35, ha="right")
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def plot_metrics_table(ax: Any, summary: dict, systems: List[str]) -> None:
    cols = [
        ("token_nll", "NLL", 3),
        ("token_ppl", "PPL", 1),
        ("token_acc", "Acc", 3),
        ("rmse", "RMSE", 4),
        ("mae", "MAE", 4),
        ("mr_stft_sc", "MR-STFT", 3),
        ("env_rms_corr", "EnvCorr", 3),
        ("tter_db_mae", "TTER", 2),
        ("onset_f1", "OnsetF1", 3),
    ]
    # Build a string table (no plots with numeric annotations).
    rows = []
    fad_label = ""
    for sys in systems:
        sysd = summary["systems"].get(sys, {}) or {}
        r = {"Codec": label_for_system(summary, sys)}
        for key, name, dig in cols:
            mu, sd = _mean_std(sysd.get(key))
            # token_acc/onset_f1 are fractions
            if key in {"token_acc", "onset_f1"} and math.isfinite(mu):
                mu *= 100.0
                if math.isfinite(sd):
                    sd *= 100.0
                r[name] = _fmt(mu, sd, 1)
            else:
                r[name] = _fmt(mu, sd, dig)
        fl, fv = get_fad_cell(summary, sys)
        if fl:
            fad_label = fl
            r[fl] = fv
        rows.append(r)

    df = pd.DataFrame(rows)
    # stable col ordering
    col_order = ["Codec"] + [c[1] for c in cols] + ([fad_label] if fad_label else [])
    df = df[[c for c in col_order if c in df.columns]]

    ax.axis("off")
    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    ax.set_title("Summary metrics (mean±std)", pad=10)


def save_dashboard(run_key: str, run_title: str, out_dir: Path) -> Path:
    set_style()
    summary, items = load_eval(out_dir)
    systems = base_systems(summary)
    n = len(systems)

    fig = plt.figure(figsize=(14.0, 7.5))
    outer = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.35)

    # Training grid (top-left)
    sub = gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=outer[0, 0], wspace=0.25)
    axs_tr = [fig.add_subplot(sub[0, i]) for i in range(n)]
    plot_training_curves(axs_tr, summary, systems)
    handles, labels = axs_tr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.065, 0.98), ncol=2)

    # Token acc heatmap (top-right)
    ax_hm = fig.add_subplot(outer[0, 1])
    plot_token_acc_heatmap(ax_hm, items, systems, summary)

    # Metrics table (bottom row, full width)
    ax_tbl = fig.add_subplot(outer[1, :])
    plot_metrics_table(ax_tbl, summary, systems)

    fig.suptitle(f"{run_title}  ({run_key})", y=0.995, fontsize=13)
    out_path = PLOTS_DIR / f"{run_key}_dashboard.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_top_worst_table(run_key: str, out_dir: Path, *, k: int = 5) -> Optional[Path]:
    summary, items = load_eval(out_dir)
    systems = base_systems(summary)
    encodec = next((s for s in systems if str(s).lower().startswith("encodec")), None)
    if encodec is None:
        return None
    enc = (
        items.loc[items["system"] == encodec]
        .groupby("kit")["token_acc"]
        .mean()
        .sort_values(ascending=True)
    )
    worst = enc.head(k)
    top = enc.tail(k).sort_values(ascending=False)
    df = pd.concat([top.rename("encodec_acc"), worst.rename("encodec_acc")], axis=0).reset_index()
    df["encodec_acc"] = (100.0 * df["encodec_acc"]).astype(float)
    df["rank_group"] = ["Top"] * len(top) + ["Worst"] * len(worst)

    # Optional per-kit fad (only if provided)
    fad_csv = PER_KIT_FAD.get(run_key)
    if fad_csv is not None and fad_csv.is_file():
        fad = pd.read_csv(fad_csv)
        fad = fad.loc[fad["system"].astype(str) == str(encodec), ["kit", "fad"]].copy()
        fad = fad.rename(columns={"fad": "encodec_fad"})
        df = df.merge(fad, on="kit", how="left")

    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 3.4))
    ax.axis("off")
    ax.set_title(f"{run_key}: Top-{k} and Worst-{k} kits by EnCodec token accuracy", pad=10)

    show_cols = ["rank_group", "kit", "encodec_acc"] + (["encodec_fad"] if "encodec_fad" in df.columns else [])
    disp = df[show_cols].copy()
    disp = disp.rename(columns={"rank_group": "Set", "kit": "Kit", "encodec_acc": "EnCodec Acc (%)", "encodec_fad": "EnCodec FAD"})
    # formatting (no overlaid numbers on plots; this is a table/matrix)
    disp["EnCodec Acc (%)"] = disp["EnCodec Acc (%)"].map(lambda x: f"{x:.2f}")
    if "EnCodec FAD" in disp.columns:
        disp["EnCodec FAD"] = disp["EnCodec FAD"].map(lambda x: "" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):.3f}")

    tbl = ax.table(cellText=disp.values.tolist(), colLabels=disp.columns.tolist(), loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)

    out_path = PLOTS_DIR / f"{run_key}_top_worst_encodec.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_collage(dash_paths: List[Tuple[str, Path]]) -> Optional[Path]:
    if len(dash_paths) != 4:
        return None
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, (title, p) in zip(axes.flat, dash_paths):
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    out = PLOTS_DIR / "dashboards_2x2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    dash_paths: List[Tuple[str, Path]] = []
    for run_key, out_dir, title in RUNS:
        if not (out_dir / "summary.json").is_file():
            print(f"[skip] missing {out_dir/'summary.json'}")
            continue
        if not (out_dir / "items.csv").is_file():
            print(f"[skip] missing {out_dir/'items.csv'}")
            continue
        print(f"[dash] {run_key} <- {out_dir}")
        dash = save_dashboard(run_key, title, out_dir)
        dash_paths.append((title, dash))

        if "all_kits" in run_key:
            tw = save_top_worst_table(run_key, out_dir, k=5)
            if tw is not None:
                print(f"[top/worst] wrote {tw}")

    coll = save_collage(dash_paths)
    if coll is not None:
        print(f"[collage] wrote {coll}")
    print(f"[done] wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()


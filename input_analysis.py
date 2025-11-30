#!/usr/bin/env python3
"""
fitts_bulk_analysis.py

Bulk-process trials_*.csv and tlx_*.csv (matching your schema) and produce:
 - Combined CSVs
 - Per-participant × mode and per-mode summaries
 - ID, Throughput, Error Rate, Learning curves
 - TLX per-mode radar and bars (ignores Overall_tlx)
 - Sound × BgMode condition heatmap & grouped bars
 - Repeated-measures ANOVA (fallback to one-way) + pairwise paired t-tests
 - PNG visualizations and CSV/text reports in ./outputs/

Usage:
  Place all trials_*.csv and tlx_*.csv in the same directory as this script (or run with a data_dir)
  Filenames must be like:
    trials_Chetan.csv
    tlx_Chetan.csv
    trials_HEMALATHA J.csv
    tlx_HEMALATHA J.csv
  Then run: python fitts_bulk_analysis.py
"""

import os
import glob
import math
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATA_DIR = Path(".")            # directory containing CSVs (change if needed)
OUT_DIR = DATA_DIR / "outputs"  # output folder
OUT_DIR.mkdir(exist_ok=True)
# If your TargetSize column is radius (not width), set radius_flag = True -> W = 2 * TargetSize
radius_flag = False
# ------------------------------------------------

# Expected trials header (case-insensitive): Name, Age, Gender, Block, Set, Mode, Trial,
# TargetSize, Distance, TimeTaken_ms, Error, ClickX, ClickY, Sound, BgMode
TRIALS_GLOB = str(DATA_DIR / "trials_*.csv")
TLX_GLOB = str(DATA_DIR / "tlx_*.csv")

def read_trials_file(path):
    df = pd.read_csv(path)
    # Normalize header names: strip whitespace
    df.columns = [c.strip() for c in df.columns]
    # Map canonical headers if variations exist (case-insensitive)
    header_map = {}
    expected = ["Name","Age","Gender","Block","Set","Mode","Trial","TargetSize","Distance",
                "TimeTaken_ms","Error","ClickX","ClickY","Sound","BgMode"]
    for col in df.columns:
        for e in expected:
            if col.strip().lower() == e.lower():
                header_map[col] = e
                break
    df = df.rename(columns=header_map)
    df["_source_file"] = Path(path).name
    return df

def read_tlx_file(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Drop Overall_tlx column if present (user said it's "Computed later" & useless)
    df = df[[c for c in df.columns if c.lower() != "overall_tlx"]]
    df["_source_file"] = Path(path).name
    return df

def compute_id(D, W):
    try:
        if pd.notnull(D) and pd.notnull(W) and W > 0:
            return math.log2(D / W + 1)
    except:
        pass
    return np.nan

def main():
    trials_files = sorted(glob.glob(TRIALS_GLOB))
    tlx_files = sorted(glob.glob(TLX_GLOB))

    if not trials_files:
        print("No trials_*.csv files found in", DATA_DIR.resolve())
        return

    print(f"Found {len(trials_files)} trial files and {len(tlx_files)} TLX files.")

    # Read all trial files and combine
    trial_dfs = []
    for p in trials_files:
        try:
            trial_dfs.append(read_trials_file(p))
        except Exception as e:
            print("Error reading", p, ":", e)
    trials = pd.concat(trial_dfs, ignore_index=True, sort=False)

    # Canonicalize columns based on the expected schema
    # Participant id from Name
    if "Name" not in trials.columns:
        # try case-insensitive match
        for c in trials.columns:
            if c.strip().lower() == "name":
                trials = trials.rename(columns={c: "Name"})
                break
    trials["participant"] = trials["Name"].astype(str).str.strip()

    # Mode
    trials["mode"] = trials["Mode"].astype(str).str.strip() if "Mode" in trials.columns else trials.get("mode", "unknown")

    # Trial index
    trials["trial"] = pd.to_numeric(trials["Trial"], errors='coerce') if "Trial" in trials.columns else trials.groupby(["participant","mode"]).cumcount()+1

    # Time (ms) -> seconds
    if "TimeTaken_ms" not in trials.columns:
        # try case-insensitive fallback for any time-like column
        for c in trials.columns:
            if "time" in c.lower():
                trials = trials.rename(columns={c: "TimeTaken_ms"})
                break
    trials["time_s"] = pd.to_numeric(trials["TimeTaken_ms"], errors='coerce')/1000.0

    # Distance and TargetSize numeric
    trials["distance"] = pd.to_numeric(trials["Distance"], errors='coerce') if "Distance" in trials.columns else np.nan
    trials["target_size"] = pd.to_numeric(trials["TargetSize"], errors='coerce') if "TargetSize" in trials.columns else np.nan

    # Error normalization (assume 1=error, 0=hit)
    if "Error" in trials.columns:
        trials["error"] = pd.to_numeric(trials["Error"], errors='coerce')
    else:
        trials["error"] = np.nan

    # Sound & BgMode normalization
    trials["sound"] = trials["Sound"].astype(str).str.lower().replace({"nan":"unknown","true":"on","false":"off","1":"on","0":"off"}) if "Sound" in trials.columns else "unknown"
    trials["bgmode"] = trials["BgMode"].astype(str).str.lower().replace({"nan":"unknown"}) if "BgMode" in trials.columns else "unknown"

    # Compute W
    if radius_flag:
        trials["W"] = trials["target_size"] * 2
    else:
        trials["W"] = trials["target_size"]

    # Compute ID and throughput per trial
    trials["ID"] = trials.apply(lambda r: compute_id(r["distance"], r["W"]), axis=1)
    trials["throughput"] = trials.apply(lambda r: (r["ID"] / r["time_s"]) if (pd.notnull(r["ID"]) and pd.notnull(r["time_s"]) and r["time_s"]>0) else np.nan, axis=1)

    # Save combined trials
    trials.to_csv(OUT_DIR / "trials_combined.csv", index=False)
    print("Saved trials_combined.csv")

    # -------- Core performance metrics --------
    per_participant_mode = trials.groupby(["participant","mode"]).agg(
        n_trials = ("trial","count"),
        mean_time_s = ("time_s","mean"),
        median_time_s = ("time_s","median"),
        mean_ID = ("ID","mean"),
        mean_throughput = ("throughput","mean"),
        error_count = ("error", lambda x: int(x.dropna().sum()) if x.dropna().size>0 else 0),
        error_rate = ("error", lambda x: np.nan if x.dropna().size==0 else x.dropna().mean())
    ).reset_index()

    per_mode = per_participant_mode.groupby("mode").agg(
        participants = ("participant","nunique"),
        total_trials = ("n_trials","sum"),
        mean_time_s = ("mean_time_s","mean"),
        median_time_s = ("median_time_s","median"),
        mean_ID = ("mean_ID","mean"),
        mean_throughput = ("mean_throughput","mean"),
        error_rate = ("error_rate","mean")
    ).reset_index()

    per_participant_mode.to_csv(OUT_DIR / "per_participant_mode_summary.csv", index=False)
    per_mode.to_csv(OUT_DIR / "per_mode_summary.csv", index=False)
    print("Saved per-participant and per-mode summaries.")

    # -------- Visualizations --------
    plt.rcParams.update({'figure.max_open_warning': 0})
    # 1) Avg Time per Mode
    plt.figure(figsize=(7,5))
    plt.bar(per_mode["mode"], per_mode["mean_time_s"])
    plt.title("Avg Time per Mode (s)")
    plt.ylabel("Time (s)")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_avg_time_per_mode.png")
    plt.close()

    # 2) Throughput per Mode
    plt.figure(figsize=(7,5))
    plt.bar(per_mode["mode"], per_mode["mean_throughput"])
    plt.title("Mean Throughput per Mode (bits/s)")
    plt.ylabel("Throughput (bits/s)")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_throughput_per_mode.png")
    plt.close()

    # 3) Error Rate per Mode
    plt.figure(figsize=(7,5))
    plt.bar(per_mode["mode"], per_mode["error_rate"])
    plt.title("Error Rate per Mode")
    plt.ylabel("Error Rate (proportion)")
    plt.xlabel("Mode")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_error_rate_per_mode.png")
    plt.close()

    # 4) Scatter: ID vs Time with linear fit per mode
    plt.figure(figsize=(8,6))
    modes = trials["mode"].unique()
    for m in modes:
        sub = trials[trials["mode"]==m][["ID","time_s"]].dropna()
        if sub.shape[0] == 0: 
            continue
        plt.scatter(sub["ID"], sub["time_s"], alpha=0.6, label=str(m))
        if len(sub) >= 2:
            coef = np.polyfit(sub["ID"], sub["time_s"], 1)
            xs = np.linspace(sub["ID"].min(), sub["ID"].max(), 100)
            plt.plot(xs, coef[0]*xs + coef[1])
    plt.xlabel("Index of Difficulty (ID)")
    plt.ylabel("Movement Time (s)")
    plt.title("ID vs Time with linear fits per mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_id_vs_time.png")
    plt.close()

    # 5) Learning curves
    lc = trials.groupby(["mode","trial"]).agg(mean_time_s = ("time_s","mean")).reset_index()
    plt.figure(figsize=(9,6))
    for m in lc["mode"].unique():
        s = lc[lc["mode"]==m].sort_values("trial")
        plt.plot(s["trial"], s["mean_time_s"], marker="o", label=str(m))
    plt.title("Learning Curve: Trial Number vs Mean Time (s)")
    plt.xlabel("Trial Number")
    plt.ylabel("Mean Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_learning_curves.png")
    plt.close()

    # -------- TLX processing (ignore Overall_tlx) ----------
    if tlx_files:
        tlx_dfs = []
        for p in tlx_files:
            try:
                tlx_dfs.append(read_tlx_file(p))
            except Exception as e:
                print("Failed reading TLX file", p, e)
        tlx_all = pd.concat(tlx_dfs, ignore_index=True, sort=False)
        # Map participant id
        pid_col = None
        for c in tlx_all.columns:
            if c.strip().lower() == "participantid":
                pid_col = c
                break
        if pid_col:
            tlx_all["participant"] = tlx_all[pid_col].astype(str).str.strip()
        else:
            tlx_all["participant"] = tlx_all["_source_file"].str.replace(r"tlx_","").str.replace(r"\.csv","", regex=True)

        # Detect TLX subscales (names expected: Mental, Physical, Temporal, Performance, Effort, Frustration)
        subs_expected = ["Mental","Physical","Temporal","Performance","Effort","Frustration"]
        subs = [s for s in subs_expected if s in tlx_all.columns]
        if not subs:
            # fallback to numeric columns except participant and _source_file
            numeric_cols = tlx_all.select_dtypes(include=[np.number]).columns.tolist()
            subs = [c for c in numeric_cols if c not in ["participant"]]
        if subs:
            # If TLX contains Mode, compute per-mode; else overall
            if "Mode" in tlx_all.columns:
                tlx_mode_summary = tlx_all.groupby("Mode")[subs].mean().reset_index()
            else:
                tlx_mode_summary = tlx_all[subs].mean().to_frame().T
                tlx_mode_summary["Mode"] = "all"
            tlx_mode_summary["tlx_total"] = tlx_mode_summary[subs].mean(axis=1)
            tlx_mode_summary.to_csv(OUT_DIR / "tlx_mode_summary.csv", index=False)
            tlx_all.groupby("participant")[subs].mean().reset_index().to_csv(OUT_DIR / "tlx_per_participant.csv", index=False)

            # Radar chart
            categories = subs
            N = len(categories)
            angles = np.linspace(0, 2*math.pi, N, endpoint=False).tolist()
            angles += angles[:1]
            plt.figure(figsize=(6,6))
            ax = plt.subplot(111, polar=True)
            for _, row in tlx_mode_summary.iterrows():
                vals = row[categories].tolist()
                vals += vals[:1]
                ax.plot(angles, vals, marker="o", label=str(row.get("Mode","mode")))
                ax.fill(angles, vals, alpha=0.1)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories)
            ax.set_title("NASA-TLX per Mode (mean per dimension)")
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))
            plt.tight_layout()
            plt.savefig(OUT_DIR / "viz_tlx_radar.png")
            plt.close()
            print("Saved TLX summaries and radar.")
        else:
            print("TLX files present but TLX subscales not detected; columns:", tlx_all.columns.tolist())
    else:
        print("No TLX files found; skipping TLX analysis.")

    # -------- Conditions: Sound x BgMode ----------
    trials["sound_clean"] = trials["sound"].fillna("unknown")
    trials["bgmode_clean"] = trials["bgmode"].fillna("unknown")
    cond = trials.groupby(["sound_clean","bgmode_clean"]).agg(
        mean_time_s = ("time_s","mean"),
        mean_throughput = ("throughput","mean"),
        error_rate = ("error", lambda x: np.nan if x.dropna().size==0 else x.dropna().mean()),
        total_trials = ("trial","count")
    ).reset_index()
    cond.to_csv(OUT_DIR / "condition_summary.csv", index=False)

    # Heatmap
    pivot = cond.pivot(index="sound_clean", columns="bgmode_clean", values="mean_time_s")
    if pivot is not None and not pivot.empty:
        plt.figure(figsize=(6,4))
        im = plt.imshow(pivot.values, aspect="auto")
        plt.title("Mean Time (s) — Sound x BgMode")
        plt.xlabel("BgMode")
        plt.ylabel("Sound")
        plt.xticks(ticks=list(range(len(pivot.columns))), labels=pivot.columns)
        plt.yticks(ticks=list(range(len(pivot.index))), labels=pivot.index)
        for (i,j), val in np.ndenumerate(pivot.values):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val>np.nanmean(pivot.values) else "black")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "viz_condition_heatmap_time.png")
        plt.close()

    # Grouped bar
    labels = [f"{s}\\n{b}" for s,b in zip(cond["sound_clean"], cond["bgmode_clean"])]
    plt.figure(figsize=(9,5))
    plt.bar(labels, cond["mean_time_s"])
    plt.title("Mean Time by Condition (Sound / BgMode)")
    plt.ylabel("Mean Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "viz_condition_grouped_time.png")
    plt.close()

    # -------- Statistics: ANOVA & pairwise tests ----------
    anova_text = ""
    try:
        from statsmodels.stats.anova import AnovaRM
        wide = per_participant_mode.pivot(index="participant", columns="mode", values="mean_time_s").reset_index()
        wide_clean = wide.dropna()
        if wide_clean.shape[0] >= 2 and wide_clean.shape[1] >= 3:
            long = wide_clean.melt(id_vars=["participant"], var_name="mode", value_name="mean_time_s")
            anova = AnovaRM(long, depvar="mean_time_s", subject="participant", within=["mode"]).fit()
            anova_text = str(anova.summary())
            with open(OUT_DIR / "anova_results.txt","w") as f:
                f.write(anova_text)
            print("Saved repeated-measures ANOVA results.")
        else:
            raise Exception("Insufficient complete cases for repeated-measures ANOVA.")
    except Exception:
        try:
            groups = [group["mean_time_s"].dropna().values for name, group in per_participant_mode.groupby("mode")]
            if len(groups) >= 2:
                fval, pval = stats.f_oneway(*groups)
                anova_text = f"One-way ANOVA F={fval:.4f}, p={pval:.6f} (fallback)"
                with open(OUT_DIR / "anova_results.txt","w") as f:
                    f.write(anova_text)
                print("Saved fallback one-way ANOVA results.")
        except Exception as e:
            print("ANOVA failed:", e)

    # Pairwise paired t-tests
    pairwise = []
    modes_list = per_participant_mode["mode"].unique().tolist()
    for i in range(len(modes_list)):
        for j in range(i+1, len(modes_list)):
            a_mode = modes_list[i]
            b_mode = modes_list[j]
            a = per_participant_mode[per_participant_mode["mode"]==a_mode].set_index("participant")["mean_time_s"]
            b = per_participant_mode[per_participant_mode["mode"]==b_mode].set_index("participant")["mean_time_s"]
            common = a.index.intersection(b.index)
            if len(common) >= 2:
                avals = a.loc[common].values
                bvals = b.loc[common].values
                tstat, pval = stats.ttest_rel(avals, bvals)
                pairwise.append({"mode_a":a_mode,"mode_b":b_mode,"n_pairs":len(common),"t":tstat,"p":pval})
    pd.DataFrame(pairwise).to_csv(OUT_DIR / "pairwise_paired_ttests.csv", index=False)

    # Save final summaries
    per_participant_mode.to_csv(OUT_DIR / "per_participant_mode_summary.csv", index=False)
    per_mode.to_csv(OUT_DIR / "per_mode_summary.csv", index=False)

    # Final report
    report = []
    report.append("Fitts Bulk Analysis Report")
    report.append("==========================")
    report.append(f"Participants: {trials['participant'].nunique()}")
    report.append(f"Modes found: {', '.join(map(str, per_mode['mode'].tolist()))}")
    report.append("\nPer-mode summary:")
    report.append(per_mode.to_string(index=False))
    if anova_text:
        report.append("\nANOVA results:\n" + anova_text)
    with open(OUT_DIR / "analysis_report.txt","w") as f:
        f.write("\n".join(report))

    print("Processing complete. Outputs written to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()

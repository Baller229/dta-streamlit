# helpers_charts.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def ensure_datetime_column(df: pd.DataFrame, ts_col: str = "Timestamp") -> pd.DataFrame:
    """
    Robust Timestamp parser:
    - if values look numeric -> assume ms since epoch
    - else parse as datetime string
    """
    out = df.copy()
    if ts_col not in out.columns:
        raise ValueError(f"Missing column: {ts_col}")

    s = out[ts_col]
    s_num = pd.to_numeric(s, errors="coerce")
    num_ratio = float(s_num.notna().mean())

    if num_ratio > 0.9:
        out[ts_col] = pd.to_datetime(s_num, unit="ms", errors="coerce")
    else:
        out[ts_col] = pd.to_datetime(s, errors="coerce")

    return out


def _collect_event_windows(
    g: pd.DataFrame,
    event_positions: np.ndarray,
    value_col: str,
    n_before: int,
    n_after: int,
    direction_value: str = "all",
):
    rows = []
    values = g[value_col].to_numpy()
    for pos in event_positions:
        for off in range(-n_before, n_after + 1):
            rows.append([off, values[pos + off], direction_value])
    return rows


def plot_qual_and_rtt_around_handovers_overlaid(
    df: pd.DataFrame,
    qual_col: str = "Qual",
    rtt_col: str = "RTT_ms",
    drive_id_col: str = "drive_id",
    ts_col: str = "Timestamp",
    is_ho_col: str = "is_handover",
    direction_col: str = "Destination",
    n_before: int = 10,
    n_after: int = 5,
    figsize: tuple = (16, 5),
    bar_width: float = 0.55,
    alpha_qual: float = 0.85,
    alpha_rtt: float = 0.45,
    # --- NEW: editable text ---
    title_text: str | None = None,
    x_label: str = "offset (samples)",
    y_label_left: str | None = None,
    y_label_right: str | None = None,
    legend_qual: str = "Qual (mean)",
    legend_rtt: str = "RTT (mean)",
    # --- NEW: editable colors ---
    color_qual: str = "C0",
    color_rtt: str = "C3",
    darken_factor: float = 0.55,
):
    """
    Overlaid dual-axis bar plot around HO (prekryvajuce sa stlpce).

    Returns: fig, axes, df_curve
      df_curve columns: [direction, offset, qual_mean, rtt_mean]
    """
    missing = [c for c in [qual_col, rtt_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")

    d = df.copy()
    d = ensure_datetime_column(d, ts_col=ts_col)
    d = d.sort_values([drive_id_col, ts_col]).reset_index(drop=True)

    if is_ho_col not in d.columns:
        raise ValueError(
            f"Missing '{is_ho_col}'. Run add_is_handover_4 first.")

    if direction_col not in d.columns:
        d[direction_col] = "all"

    def _darken_color(color: str, factor: float = 0.65):
        r, g, b = mcolors.to_rgb(color)
        return (r * factor, g * factor, b * factor)

    directions = sorted(d[direction_col].dropna().unique(), key=str)
    if len(directions) == 0:
        directions = ["all"]

    fig, axes = plt.subplots(1, len(directions), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    all_rows = []

    QUAL_BOTTOM, QUAL_TOP = -15, -12
    RTT_BOTTOM, RTT_TOP = 50, 200

    base_qual = color_qual
    base_rtt = color_rtt
    dark_qual = _darken_color(base_qual, darken_factor)
    dark_rtt = _darken_color(base_rtt, darken_factor)

    # defaults for labels
    if y_label_left is None:
        y_label_left = qual_col
    if y_label_right is None:
        y_label_right = rtt_col

    for ax, direc in zip(axes, directions):
        sub = d[d[direction_col] == direc]
        direction_title = str(direc)

        qual_rows = []
        rtt_rows = []

        for _, g in sub.groupby(drive_id_col):
            g = g.sort_values(ts_col).reset_index(drop=True)

            ho_pos = np.where(g[is_ho_col].to_numpy() == 1)[0]
            if ho_pos.size == 0:
                continue

            valid_ho = ho_pos[(ho_pos - n_before >= 0) &
                              (ho_pos + n_after < len(g))]
            if valid_ho.size == 0:
                continue

            qual_rows.extend(_collect_event_windows(
                g, valid_ho, qual_col, n_before, n_after, direction_value=direction_title))
            rtt_rows.extend(_collect_event_windows(
                g, valid_ho, rtt_col, n_before, n_after, direction_value=direction_title))

        qual_df = pd.DataFrame(qual_rows, columns=[
                               "offset", "value", "direction"])
        rtt_df = pd.DataFrame(
            rtt_rows, columns=["offset", "value", "direction"])

        if qual_df.empty or rtt_df.empty:
            ax.set_title(f"{direction_title} (no data)")
            ax.axvline(0, linewidth=1)
            ax.set_ylim(QUAL_BOTTOM, QUAL_TOP)
            ax.grid(True, alpha=0.3)
            continue

        qual_curve = qual_df.groupby("offset")["value"].mean().rename(
            "qual_mean").reset_index()
        rtt_curve = rtt_df.groupby("offset")["value"].mean().rename(
            "rtt_mean").reset_index()

        curve = pd.merge(qual_curve, rtt_curve, on="offset", how="inner")
        curve["direction"] = direction_title
        all_rows.append(curve)

        offsets = curve["offset"].to_numpy()

        qual_colors = [base_qual] * len(offsets)
        rtt_colors = [base_rtt] * len(offsets)

        idx0 = np.where(offsets == 0)[0]
        if idx0.size > 0:
            i0 = int(idx0[0])
            qual_colors[i0] = dark_qual
            rtt_colors[i0] = dark_rtt

        # Qual (left)
        ax.bar(
            offsets,
            curve["qual_mean"] - QUAL_BOTTOM,
            bottom=QUAL_BOTTOM,
            width=bar_width,
            color=qual_colors,
            alpha=alpha_qual,
            label=legend_qual,
            zorder=2,
        )
        ax.set_ylim(QUAL_BOTTOM, QUAL_TOP)
        ax.set_ylabel(y_label_left)

        # RTT (right)
        ax2 = ax.twinx()
        ax2.bar(
            offsets,
            curve["rtt_mean"] - RTT_BOTTOM,
            bottom=RTT_BOTTOM,
            width=bar_width,
            color=rtt_colors,
            alpha=alpha_rtt,
            label=legend_rtt,
            zorder=3,
        )
        ax2.set_ylim(RTT_BOTTOM, RTT_TOP)
        ax2.set_ylabel(y_label_right)

        ax.axvline(0, linewidth=1)
        # title per panel
        if title_text is None:
            ax.set_title(
                f"Qual & RTT around HO (overlaid) - {direction_title}")
        else:
            ax.set_title(f"{title_text} - {direction_title}")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3, zorder=0)

        # combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    df_curve = pd.concat(
        all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return fig, axes, df_curve

# =======================================================================================
# GRAF 2
# =======================================================================================


def _collect_event_windows_generic(
    g: pd.DataFrame,
    centers: np.ndarray,
    value_col: str,
    n_before: int,
    n_after: int,
    direction_value: str,
):
    """
    Returns list of rows: [offset, value, direction]
    for each center in centers, for offsets in [-n_before..n_after].
    Assumes bounds are valid.
    """
    rows = []
    vals = pd.to_numeric(g[value_col], errors="coerce").to_numpy()
    for c in centers:
        for off in range(-n_before, n_after + 1):
            rows.append([off, vals[c + off], direction_value])
    return rows


def compute_rtt_around_ho_curves_clean_control(
    df: pd.DataFrame,
    rtt_col: str = "RTT_ms",
    drive_id_col: str = "drive_id",
    ts_col: str = "Timestamp",
    is_ho_col: str = "is_handover",
    direction_col: str = "Destination",
    n_before: int = 10,
    n_after: int = 5,
    control_exclusion: int = 3,
    random_state: int = 42,
):
    """
    RTT around HO vs CLEAN control windows (no HO in whole window).
    Returns per-direction curve with mean + IQR.
    """
    if rtt_col not in df.columns:
        raise ValueError(f"Missing rtt_col='{rtt_col}'.")
    if is_ho_col not in df.columns:
        raise ValueError(f"Missing is_ho_col='{is_ho_col}'.")
    if direction_col not in df.columns:
        # fallback
        df = df.copy()
        df[direction_col] = "all"

    d = df.copy()
    d = ensure_datetime_column(d, ts_col=ts_col)
    d = d.sort_values([drive_id_col, ts_col]).reset_index(drop=True)

    d[rtt_col] = pd.to_numeric(d[rtt_col], errors="coerce")
    d = d.dropna(subset=[rtt_col]).copy()
    d[is_ho_col] = pd.to_numeric(
        d[is_ho_col], errors="coerce").fillna(0).astype(int)

    rng = np.random.RandomState(random_state)

    directions = sorted(d[direction_col].dropna().unique(), key=str)
    if len(directions) == 0:
        directions = ["all"]

    all_out = []

    for direc in directions:
        sub = d if direc == "all" else d[d[direction_col] == direc]
        title = "all" if direc == "all" else str(direc)

        ho_rows = []
        ctrl_rows = []
        n_ho_total = 0
        n_ctrl_total = 0

        for _, g in sub.groupby(drive_id_col):
            g = g.sort_values(ts_col).reset_index(drop=True)
            n = len(g)
            if n < (n_before + n_after + 1):
                continue

            ho_flags = g[is_ho_col].to_numpy()
            ho_pos = np.where(ho_flags == 1)[0]
            if ho_pos.size == 0:
                continue

            valid_ho = ho_pos[(ho_pos - n_before >= 0)
                              & (ho_pos + n_after < n)]
            if valid_ho.size == 0:
                continue

            ho_rows.extend(_collect_event_windows_generic(
                g, valid_ho, rtt_col, n_before, n_after, title))
            n_ho_total += len(valid_ho)

            # CLEAN control centers
            idx = np.arange(n)
            mask_ok = (idx - n_before >= 0) & (idx + n_after < n)
            mask_ok &= (ho_flags == 0)

            for p in valid_ho:
                a = max(0, p - control_exclusion)
                b = min(n, p + control_exclusion + 1)
                mask_ok[a:b] = False

            candidates = np.where(mask_ok)[0]
            if candidates.size == 0:
                continue

            clean_candidates = []
            for cpos in candidates:
                w0 = cpos - n_before
                w1 = cpos + n_after
                if ho_flags[w0:w1 + 1].sum() == 0:
                    clean_candidates.append(cpos)

            if len(clean_candidates) == 0:
                continue

            clean_candidates = np.array(clean_candidates, dtype=int)
            k = min(len(valid_ho), len(clean_candidates))
            ctrl_centers = rng.choice(clean_candidates, size=k, replace=False)

            ctrl_rows.extend(_collect_event_windows_generic(
                g, ctrl_centers, rtt_col, n_before, n_after, title))
            n_ctrl_total += k

        ho_df = pd.DataFrame(ho_rows, columns=["offset", "rtt", "direction"])
        ctrl_df = pd.DataFrame(ctrl_rows, columns=[
                               "offset", "rtt", "direction"])
        if ho_df.empty or ctrl_df.empty:
            continue

        def _agg(x):
            return pd.Series({
                "mean": x.mean(),
                "p25": x.quantile(0.25),
                "p50": x.quantile(0.50),
                "p75": x.quantile(0.75),
            })

        ho_curve = ho_df.groupby("offset")["rtt"].apply(_agg).unstack()
        ctrl_curve = ctrl_df.groupby("offset")["rtt"].apply(_agg).unstack()

        curve = ho_curve.add_prefix("ho_").join(
            ctrl_curve.add_prefix("ctrl_"), how="inner").reset_index()
        curve["direction"] = title
        curve["n_ho_windows"] = n_ho_total
        curve["n_ctrl_windows"] = n_ctrl_total
        all_out.append(curve)

    return pd.concat(all_out, ignore_index=True) if all_out else pd.DataFrame()


def plot_rtt_around_ho_with_iqr_by_direction(
    curve: pd.DataFrame,
    figsize=(14, 5),
    use_median: bool = False,
    # --- editable from Streamlit ---
    title_text: str = "RTT around HO vs CTRL (IQR)",
    x_label: str = "offset (samples)",
    y_label: str = "RTT_ms",
    legend_ho_center: str = "HO mean",
    legend_ctrl_center: str = "CTRL mean",
    legend_ho_iqr: str = "HO IQR",
    legend_ctrl_iqr: str = "CTRL IQR",
    color_ho: str = "#1f77b4",
    color_ctrl: str = "#d62728",
):
    """
    Subplots per direction. Plots HO vs CTRL center (mean/median) and IQR.
    """
    if curve is None or curve.empty:
        raise ValueError("curve is empty.")

    dirs = sorted(curve["direction"].dropna().unique(), key=str)
    fig, axes = plt.subplots(
        1, len(dirs), figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, direc in zip(axes, dirs):
        g = curve[curve["direction"] == direc].sort_values("offset")

        ho_center = g["ho_p50"] if use_median else g["ho_mean"]
        ctrl_center = g["ctrl_p50"] if use_median else g["ctrl_mean"]

        ho_center_label = "HO median" if use_median else legend_ho_center
        ctrl_center_label = "CTRL median" if use_median else legend_ctrl_center

        ax.plot(g["offset"], ho_center, label=ho_center_label, color=color_ho)
        ax.fill_between(g["offset"], g["ho_p25"], g["ho_p75"],
                        alpha=0.2, label=legend_ho_iqr, color=color_ho)

        ax.plot(g["offset"], ctrl_center,
                label=ctrl_center_label, color=color_ctrl)
        ax.fill_between(g["offset"], g["ctrl_p25"], g["ctrl_p75"],
                        alpha=0.2, label=legend_ctrl_iqr, color=color_ctrl)

        ax.axvline(0, linewidth=1)
        ax.grid(True, alpha=0.3)
        nho = int(g["n_ho_windows"].iloc[0])
        nct = int(g["n_ctrl_windows"].iloc[0])
        ax.set_title(f"{direc} | HO={nho} CTRL={nct}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    fig.suptitle(title_text)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    return fig, axes

# =======================================================================================
# GRAF 3
# =======================================================================================


def filter_rtt_outliers_iqr(
    df: pd.DataFrame,
    value_col: str = "RTT_ms",
    q1: float = 0.10,
    q3: float = 0.90,
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Vráti kópiu df, kde sú ponechané len riadky,
    ktorých `value_col` (RTT) leží MIMO IQR rozsahu:
      Q1 - iqr_factor * IQR až Q3 + iqr_factor * IQR.

    Na NaN hodnoty v `value_col` sa filter neaplikuje (budú vyhodené).
    Pozn.: default q1/q3 máš 0.10/0.90 (tak ako v tvojej verzii).
    """
    if value_col not in df.columns:
        raise ValueError(f"Missing column: {value_col}")

    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if s.empty:
        return df.iloc[0:0].copy()

    q1v = float(s.quantile(q1))
    q3v = float(s.quantile(q3))
    iqr = q3v - q1v

    lower = q1v - iqr_factor * iqr
    upper = q3v + iqr_factor * iqr

    v = pd.to_numeric(df[value_col], errors="coerce")
    mask_out = (v < lower) | (v > upper)
    df_outliers = df[mask_out].copy()

    df_outliers.attrs["rtt_iqr_lower"] = float(lower)
    df_outliers.attrs["rtt_iqr_upper"] = float(upper)
    df_outliers.attrs["rtt_q1"] = float(q1v)
    df_outliers.attrs["rtt_q3"] = float(q3v)
    df_outliers.attrs["rtt_iqr_factor"] = float(iqr_factor)

    return df_outliers


def plot_drive_bucket_outlier_heatmap(
    df_all: pd.DataFrame,
    df_outliers: pd.DataFrame,
    drive_id_col: str = "drive_id",
    bucket_index_col: str = "common_bucket_index",
    bucket_col: str = "bucket_name",  # zatiaľ nepoužité, nechávam kvôli kompatibilite
    figsize: tuple = (10, 6),
    title: str = "Podiel RTT outlierov podľa drive_id a bucketu",
    x_label: str = "common_bucket_index",
    y_label: str = "drive_id",
    colorbar_label: str = "ratio_outliers",
    cmap: str = "viridis",
):
    """
    Heatmapa podielu outlierov na úrovni (drive_id, bucket).

    Riadky: drive_id
    Stĺpce: common_bucket_index (bucket)
    Hodnota: ratio_outliers v danom drive_id a buckete.

    Returns: fig, ax, pivot
      pivot: index=drive_id, columns=bucket_index_col, values=ratio_outliers
    """
    for c in [drive_id_col, bucket_index_col]:
        if c not in df_all.columns:
            raise ValueError(f"Missing column in df_all: {c}")
        if c not in df_outliers.columns:
            # df_outliers môže mať menej stĺpcov ak si niečo predtým selektoval
            raise ValueError(f"Missing column in df_outliers: {c}")

    # celkové počty
    df_all_b = df_all.dropna(subset=[drive_id_col, bucket_index_col]).copy()
    total = (
        df_all_b
        .groupby([drive_id_col, bucket_index_col], as_index=False)
        .size()
        .rename(columns={"size": "count_total"})
    )

    # outliery
    df_out_b = df_outliers.dropna(
        subset=[drive_id_col, bucket_index_col]).copy()
    out = (
        df_out_b
        .groupby([drive_id_col, bucket_index_col], as_index=False)
        .size()
        .rename(columns={"size": "count_outliers"})
    )

    merged = pd.merge(
        total,
        out,
        on=[drive_id_col, bucket_index_col],
        how="left",
    )
    merged["count_outliers"] = merged["count_outliers"].fillna(0).astype(int)
    merged["ratio_outliers"] = merged["count_outliers"] / \
        merged["count_total"].replace(0, np.nan)

    pivot = merged.pivot(
        index=drive_id_col,
        columns=bucket_index_col,
        values="ratio_outliers",
    )

    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(pivot.to_numpy(), aspect="auto", origin="upper", cmap=cmap)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int), rotation=90)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    fig.colorbar(cax, ax=ax, label=colorbar_label)
    fig.tight_layout()
    return fig, ax, pivot

# =======================================================================================
# GRAF 4
# =======================================================================================


def compute_bucket_handover_stats(
    df: pd.DataFrame,
    drive_id_col: str = "drive_id",
    bucket_index_col: str = "common_bucket_index",
    bucket_col: str = "bucket_name",
    is_ho_col: str = "is_handover",
    rtt_col: str = "RTT_ms",
) -> pd.DataFrame:
    """
    Štatistiky na úrovni (drive_id, bucket):
      - n_samples, n_handovers, handover_rate
      - rtt_mean, rtt_median, rtt_std, rtt_q1, rtt_q3, rtt_iqr
    """
    if is_ho_col not in df.columns:
        raise ValueError(
            f"Stĺpec {is_ho_col} neexistuje – najprv doplň is_handover.")

    for c in [drive_id_col, bucket_index_col, bucket_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    d = df.dropna(subset=[drive_id_col, bucket_index_col, bucket_col]).copy()

    # numeric safety
    d[rtt_col] = pd.to_numeric(d[rtt_col], errors="coerce")
    d[is_ho_col] = pd.to_numeric(
        d[is_ho_col], errors="coerce").fillna(0).astype(int)

    def q1(x):
        return x.quantile(0.25)

    def q3(x):
        return x.quantile(0.75)

    group_cols = [drive_id_col, bucket_index_col, bucket_col]

    agg = (
        d.groupby(group_cols)
        .agg(
            n_samples=(rtt_col, lambda x: x.notna().sum()),
            n_handovers=(is_ho_col, "sum"),
            rtt_mean=(rtt_col, "mean"),
            rtt_median=(rtt_col, "median"),
            rtt_std=(rtt_col, "std"),
            rtt_q1=(rtt_col, q1),
            rtt_q3=(rtt_col, q3),
        )
        .reset_index()
    )

    agg["handover_rate"] = agg["n_handovers"] / \
        agg["n_samples"].replace(0, np.nan)
    agg["rtt_iqr"] = agg["rtt_q3"] - agg["rtt_q1"]

    agg = agg.sort_values([drive_id_col, bucket_index_col]
                          ).reset_index(drop=True)
    return agg


def plot_drive_bucket_handover_heatmap(
    stats_per_drive: pd.DataFrame,
    drive_id_col: str = "drive_id",
    bucket_index_col: str = "common_bucket_index",
    figsize: tuple = (10, 6),
    title: str = "Handover rate podľa drive_id a bucketu",
    x_label: str = "common_bucket_index",
    y_label: str = "drive_id",
    colorbar_label: str = "handover_rate",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Heatmapa handover_rate na úrovni (drive_id, bucket).
    Returns: fig, ax, pivot
    """
    if "handover_rate" not in stats_per_drive.columns:
        raise ValueError(
            "stats_per_drive musí obsahovať stĺpec 'handover_rate'.")

    df_plot = stats_per_drive.dropna(subset=["handover_rate"]).copy()
    if df_plot.empty:
        raise ValueError(
            "Žiadne dáta na vykreslenie (handover_rate je prázdny).")

    pivot = df_plot.pivot(
        index=drive_id_col,
        columns=bucket_index_col,
        values="handover_rate",
    )

    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    A = pivot.values.astype(float)

    # default vmin/vmax
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        if np.all(np.isnan(A)):
            vmax = 1.0
        else:
            vmax_val = np.nanmax(A)
            if not np.isfinite(vmax_val) or vmax_val == 0:
                vmax_val = 1.0
            vmax = float(vmax_val)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    im = ax.imshow(
        A,
        aspect="auto",
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(d) for d in pivot.index])

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([int(b) for b in pivot.columns], rotation=90)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(colorbar_label)

    return fig, ax, pivot

# =======================================================================================
# GRAF 5
# =======================================================================================


def plot_rtt_vs_handover_binned_box(
    stats_df: pd.DataFrame,
    ho_col: str = "handover_rate",
    rtt_col: str = "rtt_mean",
    n_bins: int = 6,
    figsize: tuple = (8, 4),
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    box_facecolor: str = "lightgray",
    box_alpha: float = 0.7,
    show_fliers: bool = True,
    annotate_counts: bool = True,
):
    """
    Binned boxplot: RTT vs handover_rate bez scatteru.

    Returns: fig, ax, bin_info
    """
    for c in [ho_col, rtt_col]:
        if c not in stats_df.columns:
            raise ValueError(f"Missing column in stats_df: {c}")

    d = stats_df[[ho_col, rtt_col]].dropna().copy()
    if d.empty:
        raise ValueError("No data to plot.")

    d[ho_col] = pd.to_numeric(d[ho_col], errors="coerce")
    d[rtt_col] = pd.to_numeric(d[rtt_col], errors="coerce")
    d = d.dropna(subset=[ho_col, rtt_col]).copy()

    d = d[d[ho_col] >= 0]
    if d.empty:
        raise ValueError("No non-negative handover_rate values to plot.")

    ho_min = float(d[ho_col].min())
    ho_max = float(d[ho_col].max())

    if ho_max == ho_min:
        raise ValueError("handover_rate has no variation (single value).")

    bins = np.linspace(ho_min, ho_max, n_bins + 1)
    bin_labels = [f"{bins[i]:.3f}-{bins[i+1]:.3f}" for i in range(n_bins)]

    d["ho_bin"] = pd.cut(
        d[ho_col],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )

    d = d.dropna(subset=["ho_bin"])
    if d.empty:
        raise ValueError("All bins are empty after binning.")

    rtt_groups = []
    valid_labels = []
    ho_bin_min = []
    ho_bin_max = []
    ho_bin_count = []
    ho_bin_median_rtt = []

    for i, label in enumerate(bin_labels):
        g = d[d["ho_bin"] == label][rtt_col]
        if g.empty:
            continue
        arr = g.to_numpy()
        rtt_groups.append(arr)
        valid_labels.append(label)

        ho_bin_min.append(float(bins[i]))
        ho_bin_max.append(float(bins[i + 1]))
        ho_bin_count.append(int(arr.shape[0]))
        ho_bin_median_rtt.append(float(np.median(arr)))

    if not rtt_groups:
        raise ValueError("No non-empty bins to plot.")

    bin_info = pd.DataFrame(
        {
            "bin_label": valid_labels,
            "ho_min": ho_bin_min,
            "ho_max": ho_bin_max,
            "count": ho_bin_count,
            "rtt_median": ho_bin_median_rtt,
        }
    )

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        rtt_groups,
        labels=valid_labels,
        showfliers=show_fliers,
        patch_artist=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(box_facecolor)
        patch.set_alpha(box_alpha)

    if x_label is None:
        x_label = f"{ho_col} bins"
    if y_label is None:
        y_label = rtt_col
    if title is None:
        title = f"{rtt_col} vs {ho_col} (binned boxplots)"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if annotate_counts:
        y_top = ax.get_ylim()[1]
        for i, cnt in enumerate(ho_bin_count, start=1):
            ax.text(i, y_top * 0.98, str(cnt),
                    ha="center", va="top", fontsize=7)

    fig.tight_layout()
    return fig, ax, bin_info

# ============================================================================================
# GRAF 6
# ============================================================================================


def plot_rtt_boxplots_by_direction_and_bucket(
    df: pd.DataFrame,
    value_col: str = "RTT_ms",
    direction_col: str = "Destination",
    bucket_col: str = "bucket_name",
    bucket_index_col: str = "common_bucket_index",
    filter_outliers: bool = False,
    use_log_scale: bool = False,
    figsize: tuple = (14, 5),
    sharey: bool = True,
    # --- editable ---
    title: str | None = None,
    x_label: str = "bucket",
    y_label: str | None = None,
    box_facecolor: str = "lightgray",
    box_alpha: float = 0.7,
    show_fliers: bool = True,
    # outlier params (keď filter_outliers=True)
    out_q1: float = 20.0,
    out_q3: float = 80.0,
    out_iqr_factor: float = 1.5,
):
    """
    Subplot per direction. Boxplot RTT per bucket (globálne cez všetky drive_id).

    Returns: fig, axes
    """
    needed_cols = [value_col, direction_col, bucket_col, bucket_index_col]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    d = df.dropna(subset=needed_cols).copy()
    if d.empty:
        raise ValueError("No data to plot (after dropping NaN).")

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col]).copy()

    d = d[d[value_col] > 0]
    if d.empty:
        raise ValueError("No positive RTT values to plot.")

    directions = sorted(d[direction_col].dropna().unique(), key=str)
    n_dirs = len(directions)
    if n_dirs == 0:
        raise ValueError("No directions to plot.")

    fig, axes = plt.subplots(1, n_dirs, figsize=figsize, squeeze=False)
    axes = axes[0]

    global_min = float(d[value_col].min())
    global_max = float(d[value_col].max())

    if y_label is None:
        y_label = value_col
    if title is None:
        title = f"{value_col} per bucket by {direction_col}"

    for ax, direc in zip(axes, directions):
        sub_d = d[d[direction_col] == direc].copy()
        if sub_d.empty:
            ax.set_title(f"{direction_col}={direc} (no data)")
            continue

        buckets_order = (
            sub_d[[bucket_index_col, bucket_col]]
            .drop_duplicates()
            .sort_values(bucket_index_col)
        )
        bucket_labels = buckets_order[bucket_col].astype(str).tolist()

        groups = []
        for _, row in buckets_order.iterrows():
            bname = row[bucket_col]
            vals = sub_d.loc[sub_d[bucket_col] == bname,
                             value_col].to_numpy().astype(float)

            if vals.size == 0:
                groups.append(np.array([]))
                continue

            if filter_outliers:
                q1v = np.percentile(vals, out_q1)
                q3v = np.percentile(vals, out_q3)
                iqr = q3v - q1v
                lo = q1v - out_iqr_factor * iqr
                hi = q3v + out_iqr_factor * iqr
                vals = vals[(vals >= lo) & (vals <= hi)]

            groups.append(vals)

        non_empty_groups, non_empty_labels = [], []
        for g, lab in zip(groups, bucket_labels):
            if g is not None and len(g) > 0:
                non_empty_groups.append(g)
                non_empty_labels.append(lab)

        if not non_empty_groups:
            ax.set_title(f"{direction_col}={direc} (no data after filtering)")
            continue

        bp = ax.boxplot(
            non_empty_groups,
            labels=non_empty_labels,
            showfliers=show_fliers,
            patch_artist=True,
        )

        for patch in bp["boxes"]:
            patch.set_facecolor(box_facecolor)
            patch.set_alpha(box_alpha)

        ax.set_title(str(direc))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticklabels(non_empty_labels, rotation=90)

        if use_log_scale:
            ax.set_yscale("log")

        if sharey:
            ax.set_ylim(global_min * 0.95, global_max * 1.05)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes

# =======================================================================================
# GRAF 7
# =======================================================================================


def _to_local_xy(lat: np.ndarray, lon: np.ndarray, lat0: float | None = None, lon0: float | None = None):
    """
    Jednoduchá lokálna equirectangular projekcia do metrov.
    Vracia: x[m], y[m], lat0, lon0 (referenčný bod)
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    if lat0 is None:
        lat0 = float(np.nanmedian(lat))
    if lon0 is None:
        lon0 = float(np.nanmedian(lon))

    R = 6371000.0  # m
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    x = (lon_rad - lon0_rad) * np.cos(lat0_rad) * R
    y = (lat_rad - lat0_rad) * R
    return x, y, lat0, lon0


def _local_density_counts(coords: np.ndarray, radius_m: float) -> np.ndarray:
    """
    Vráti počet susedov (vrátane seba) v okruhu radius_m pre každý bod.
    Preferuje cKDTree (scipy), inak fallback brute-force.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    try:
        from scipy.spatial import cKDTree as KDTree  # najrýchlejšie
        tree = KDTree(coords)
        neighbors = tree.query_ball_point(coords, r=radius_m)
        return np.array([len(idx) for idx in neighbors], dtype=float)
    except Exception:
        # fallback O(n^2) – OK pre menšie počty outlierov
        d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        return np.sum(d2 <= (radius_m ** 2), axis=1).astype(float)


def plot_rtt_outlier_cloud_by_direction(
    df_outliers: pd.DataFrame,
    df_all: pd.DataFrame,
    drive_id_col: str = "drive_id",
    direction_col: str = "Destination",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    ts_col: str = "Timestamp",
    radius_m: float = 150.0,
    figsize: tuple = (18, 7),
    # --- editable ---
    title_prefix: str = "Outlier cloud",
    cmap_name: str = "YlOrRd",
    density_label: str = "relative local density",
    ref_label: str = "ref drive",
    start_label: str = "Start",
    end_label: str = "End",
    ref_color: str = "lightgray",
    ref_lw: float = 2.0,
    ref_alpha: float = 0.7,
    start_marker: str = "^",
    end_marker: str = "x",
    start_end_size: int = 80,
    outlier_size: int = 40,
    outlier_alpha: float = 0.95,
):
    """
    Outlier cloud podľa smeru:
    - subplot per direction
    - referenčný drive = najdlhší drive v smere
    - outliery zafarbené podľa lokálnej hustoty (počet susedov v okruhu radius_m)
    """
    if df_all is None or df_outliers is None:
        raise ValueError("df_all a df_outliers nesmu byt None.")

    for c in [direction_col, lat_col, lon_col]:
        if c not in df_outliers.columns:
            raise ValueError(f"Missing column in df_outliers: {c}")
        if c not in df_all.columns:
            raise ValueError(f"Missing column in df_all: {c}")

    df_o = df_outliers.dropna(subset=[lat_col, lon_col, direction_col]).copy()
    if df_o.empty:
        raise ValueError(
            "V df_outliers nie su ziadne data s GPS na vykreslenie.")

    directions = sorted(df_o[direction_col].dropna().unique(), key=str)
    if len(directions) == 0:
        raise ValueError("V df_outliers nie su ziadne smery na vykreslenie.")

    fig, axes = plt.subplots(
        1, len(directions), figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax in axes:
        ax.set_visible(False)

    cmap = plt.get_cmap(cmap_name)

    for i, direc in enumerate(directions):
        ax = axes[i]
        ax.set_visible(True)

        d_all_dir = df_all[
            (df_all[direction_col] == direc)
            & df_all[lat_col].notna()
            & df_all[lon_col].notna()
        ].copy()

        if d_all_dir.empty:
            ax.set_title(f"{direction_col}={direc} (no data)")
            continue

        # referenčný drive = najviac vzoriek
        if drive_id_col not in d_all_dir.columns:
            raise ValueError(f"Missing drive_id_col in df_all: {drive_id_col}")

        counts = d_all_dir.groupby(drive_id_col).size()
        ref_id = counts.idxmax()

        ref = d_all_dir[d_all_dir[drive_id_col] == ref_id].copy()
        if ts_col in ref.columns:
            # Timestamp môže byť string/numeric – triedime len ak sa dá
            try:
                ref = ensure_datetime_column(
                    ref, ts_col=ts_col).sort_values(ts_col)
            except Exception:
                ref = ref.sort_values(ts_col)

        lat_ref = ref[lat_col].to_numpy()
        lon_ref = ref[lon_col].to_numpy()

        x_ref, y_ref, lat0, lon0 = _to_local_xy(lat_ref, lon_ref)

        ax.plot(
            x_ref, y_ref,
            color=ref_color,
            linewidth=ref_lw,
            alpha=ref_alpha,
            label=ref_label,
            zorder=1,
        )
        ax.scatter(x_ref[0], y_ref[0], marker=start_marker,
                   s=start_end_size, color="black", label=start_label, zorder=3)
        ax.scatter(x_ref[-1], y_ref[-1], marker=end_marker,
                   s=start_end_size, color="black", label=end_label, zorder=3)

        # outliery v smere
        d_o_dir = df_o[df_o[direction_col] == direc].dropna(
            subset=[lat_col, lon_col]).copy()
        if d_o_dir.empty:
            ax.set_title(f"{direc} (bez outlierov)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            ax.legend(fontsize=8, loc="best")
            continue

        lat_o = d_o_dir[lat_col].to_numpy()
        lon_o = d_o_dir[lon_col].to_numpy()

        x_o, y_o, _, _ = _to_local_xy(lat_o, lon_o, lat0, lon0)
        coords = np.column_stack([x_o, y_o])

        density = _local_density_counts(coords, radius_m=radius_m)

        d_max = float(density.max()) if density.size else 0.0
        dens_norm = (density / d_max) if d_max > 0 else density

        sc = ax.scatter(
            x_o, y_o,
            c=dens_norm,
            cmap=cmap,
            s=outlier_size,
            alpha=outlier_alpha,
            edgecolors="none",
            zorder=2,
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{direc} – {title_prefix}")
        ax.legend(fontsize=8, loc="best")

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(density_label)

    fig.tight_layout()
    return fig, axes

# =======================================================================================
# GRAF 8
# =======================================================================================


def compute_cellid_mixing_for_heatmaps(
    df: pd.DataFrame,
    drive_id_col: str = "drive_id",
    bucket_index_col: str = "common_bucket_index",
    cell_col: str = "CellID",
):
    """
    Prepare per-(drive_id, bucket_index) stats for CellID mixing heatmaps
    s tym, ze buckety su spolocne pre oba smery (pouziva sa len common_bucket_index).

    For each (drive_id, bucket_index) it computes:
      - dominant_cellid : CellID with highest sample count
      - num_cellids     : number of distinct CellID values

    Returns:
      mixing_df          : long format table
      dominant_pivot     : pivot [drive_id x bucket_index] with dominant_cellid (object)
      num_cellids_pivot  : pivot [drive_id x bucket_index] with number of distinct CellIDs (int)
    """
    d = df.dropna(subset=[drive_id_col, bucket_index_col, cell_col]).copy()
    if d.empty:
        raise ValueError(
            "No data with drive_id, bucket_index and CellID to compute mixing.")

    # base group: (drive, bucket_index, CellID) -> count
    g = (
        d.groupby([drive_id_col, bucket_index_col, cell_col])
        .size()
        .reset_index(name="count")
    )

    group_keys = [drive_id_col, bucket_index_col]

    # dominant CellID per (drive, bucket_index)
    idx_dom = g.groupby(group_keys)["count"].idxmax()
    dom = g.loc[idx_dom, group_keys + [cell_col]].rename(
        columns={cell_col: "dominant_cellid"}
    )

    # number of distinct CellIDs per (drive, bucket_index)
    nunq = (
        g.groupby(group_keys)[cell_col]
        .nunique()
        .reset_index(name="num_cellids")
    )

    mixing_df = dom.merge(nunq, on=group_keys, how="inner")

    # poradie drive_id a bucket_index
    drives = sorted(mixing_df[drive_id_col].unique())
    buckets = sorted(mixing_df[bucket_index_col].unique())

    # pivoty: index = drive_id, columns = bucket_index
    dominant_pivot = pd.DataFrame(
        index=drives,
        columns=buckets,
        dtype=object,
    )

    num_cellids_pivot = pd.DataFrame(
        index=drives,
        columns=buckets,
        dtype=float,
    )

    for _, row in mixing_df.iterrows():
        did = row[drive_id_col]
        bidx = row[bucket_index_col]
        dom_cell = row["dominant_cellid"]
        n_cells = row["num_cellids"]

        if did in dominant_pivot.index and bidx in dominant_pivot.columns:
            dominant_pivot.loc[did, bidx] = dom_cell
            num_cellids_pivot.loc[did, bidx] = n_cells

    return mixing_df, dominant_pivot, num_cellids_pivot


def plot_cellid_mixing_heatmaps(
    dominant_pivot: pd.DataFrame,
    num_cellids_pivot: pd.DataFrame,
    figsize: tuple = (16, 6),
    # --- LEFT (dominant CellID) editable ---
    title_left: str = "Dominant CellID per bucket/drive",
    xlabel_left: str = "bucket index (1..N)",
    ylabel_left: str = "drive_id",
    cmap_left: str = "plasma",
    cbar_label_left: str = "dominant CellID",
    # --- RIGHT (num distinct CellIDs) editable ---
    title_right: str = "Number of distinct CellIDs per bucket/drive",
    xlabel_right: str = "bucket index (1..N)",
    ylabel_right: str = "drive_id",
    cmap_right: str = "Blues",
    cbar_label_right: str = "num_cellids",
    # --- shared / misc ---
    bad_color: str = "lightgrey",
    x_labels_plus_one: bool = True,
):
    """
    2 heatmapy vedľa seba:
      LEFT  = dominant CellID (kategórie mapované na 1..K, gradient)
      RIGHT = počet unikátnych CellID (diskrétne bloky v colorbare)
    """
    if dominant_pivot.shape != num_cellids_pivot.shape:
        raise ValueError(
            "dominant_pivot and num_cellids_pivot must have the same shape.")

    # --------- LEFT: dominant CellID ---------
    dom_values = dominant_pivot.values

    unique_cellids = sorted({v for v in dom_values.ravel() if pd.notna(v)})
    if not unique_cellids:
        raise ValueError("No dominant CellID values to plot.")

    n_cells = len(unique_cellids)
    cellid_to_val = {cid: i + 1 for i, cid in enumerate(unique_cellids)}

    H_dom = np.full(dom_values.shape, np.nan, dtype=float)
    for i in range(dom_values.shape[0]):
        for j in range(dom_values.shape[1]):
            v = dom_values[i, j]
            if pd.notna(v) and v in cellid_to_val:
                H_dom[i, j] = cellid_to_val[v]

    base_cmap_dom = plt.get_cmap(cmap_left)
    cmap_cells = base_cmap_dom.copy()
    cmap_cells.set_bad(color=bad_color)

    H_dom_masked = np.ma.masked_invalid(H_dom)
    vmin_dom, vmax_dom = 1, n_cells

    # --------- RIGHT: num_cellids (discrete blocks) ---------
    H_cnt = num_cellids_pivot.values.astype(float)

    base_cmap_cnt = plt.get_cmap(cmap_right)
    cmap_cnt = base_cmap_cnt.copy()
    cmap_cnt.set_bad(color=bad_color)

    valid_vals = H_cnt[np.isfinite(H_cnt)]
    if valid_vals.size > 0:
        min_count = int(valid_vals.min())
        max_count = int(valid_vals.max())
    else:
        min_count, max_count = 0, 1

    if min_count == max_count:
        max_count = min_count + 1

    bounds = np.arange(min_count - 0.5, max_count + 1.5, 1.0)
    norm_cnt = mcolors.BoundaryNorm(bounds, cmap_cnt.N)

    H_cnt_masked = np.ma.masked_invalid(H_cnt)

    # --------- labels ---------
    bucket_indices = list(dominant_pivot.columns)

    if x_labels_plus_one:
        try:
            x_labels = [str(int(b) + 1) for b in bucket_indices]
        except Exception:
            x_labels = [str(b) for b in bucket_indices]
    else:
        x_labels = [str(b) for b in bucket_indices]

    y_labels = list(dominant_pivot.index.astype(str))

    # --------- plot ---------
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # LEFT
    ax0 = axes[0]
    im0 = ax0.imshow(
        H_dom_masked,
        aspect="auto",
        cmap=cmap_cells,
        vmin=vmin_dom,
        vmax=vmax_dom,
        interpolation="nearest",
    )
    ax0.set_title(title_left)
    ax0.set_xlabel(xlabel_left)
    ax0.set_ylabel(ylabel_left)
    ax0.set_xticks(np.arange(len(x_labels)))
    ax0.set_xticklabels(x_labels, rotation=90)
    ax0.set_yticks(np.arange(len(y_labels)))
    ax0.set_yticklabels(y_labels)

    cbar0 = fig.colorbar(im0, ax=ax0)
    cbar0.set_label(cbar_label_left)
    ticks_dom = np.arange(1, n_cells + 1)
    cbar0.set_ticks(ticks_dom)
    cbar0.set_ticklabels([str(cid) for cid in unique_cellids])
    cbar0.ax.tick_params(labelsize=6)

    # RIGHT
    ax1 = axes[1]
    im1 = ax1.imshow(
        H_cnt_masked,
        aspect="auto",
        cmap=cmap_cnt,
        norm=norm_cnt,
        interpolation="nearest",
    )
    ax1.set_title(title_right)
    ax1.set_xlabel(xlabel_right)
    ax1.set_ylabel(ylabel_right)
    ax1.set_xticks(np.arange(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=90)
    ax1.set_yticks(np.arange(len(y_labels)))
    ax1.set_yticklabels(y_labels)

    cbar1 = fig.colorbar(im1, ax=ax1, boundaries=bounds)
    cbar1.set_label(cbar_label_right)
    ticks_cnt = np.arange(min_count, max_count + 1)
    cbar1.set_ticks(ticks_cnt)
    cbar1.set_ticklabels([str(t) for t in ticks_cnt])

    fig.tight_layout()
    return fig, axes

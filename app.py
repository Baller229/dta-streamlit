# app.py
from __future__ import annotations

import io
from datetime import datetime

import streamlit as st
import pandas as pd

from helpers_charts import (
    plot_qual_and_rtt_around_handovers_overlaid,
    compute_rtt_around_ho_curves_clean_control,
    plot_rtt_around_ho_with_iqr_by_direction,
    filter_rtt_outliers_iqr,
    plot_drive_bucket_outlier_heatmap,
    compute_bucket_handover_stats,
    plot_drive_bucket_handover_heatmap,
    plot_rtt_vs_handover_binned_box,
    plot_rtt_boxplots_by_direction_and_bucket,
    plot_rtt_outlier_cloud_by_direction,
    compute_cellid_mixing_for_heatmaps,
    plot_cellid_mixing_heatmaps,
)

DEFAULT_CSV_PATH = "dta.csv"


def _export_figure_buttons(fig, base_name: str, key_prefix: str):
    """
    Renderuje download buttons pre SVG/PNG/PDF pre matplotlib fig.
    base_name: napr. "graf_1_qual_rtt"
    key_prefix: unikátny prefix na Streamlit key (napr. "g1")
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{base_name}_{ts}"

    c1, c2, c3 = st.columns(3)

    # SVG
    svg_buf = io.BytesIO()
    fig.savefig(svg_buf, format="svg", bbox_inches="tight")
    svg_buf.seek(0)
    with c1:
        st.download_button(
            label="Export to SVG",
            data=svg_buf,
            file_name=f"{fname}.svg",
            mime="image/svg+xml",
            key=f"{key_prefix}_dl_svg",
        )

    # PNG
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    png_buf.seek(0)
    with c2:
        st.download_button(
            label="Export to PNG",
            data=png_buf,
            file_name=f"{fname}.png",
            mime="image/png",
            key=f"{key_prefix}_dl_png",
        )

    # PDF
    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)
    with c3:
        st.download_button(
            label="Export to PDF",
            data=pdf_buf,
            file_name=f"{fname}.pdf",
            mime="application/pdf",
            key=f"{key_prefix}_dl_pdf",
        )


@st.cache_data
def _load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


st.set_page_config(page_title="Drive-test grafy", layout="wide")
st.title("Drive-test grafy (CSV upload)")

st.subheader("Vstupné dáta")

uploaded = st.file_uploader(
    "Nahraj vlastný CSV (voliteľné)",
    type=["csv"],
    key="csv_upload",
)

if uploaded is not None:
    try:
        df = _load_csv_from_upload(uploaded)
        st.success("Použité: uploadnutý CSV súbor")
    except Exception as e:
        st.error(f"Nepodarilo sa načítať uploadnutý CSV: {e}")
        st.stop()
else:
    try:
        df = _load_csv_from_path(DEFAULT_CSV_PATH)
        st.info(f"Použité: default CSV ({DEFAULT_CSV_PATH})")
    except Exception as e:
        st.error(
            f"Nepodarilo sa načítať default CSV ({DEFAULT_CSV_PATH}). "
            f"Pridaj ho vedľa app.py alebo nahraj CSV ručne.\n\n{e}"
        )
        st.stop()

st.success(f"Načítané: {df.shape[0]:,} riadkov × {df.shape[1]} stĺpcov")

with st.expander("Preview dát"):
    st.dataframe(df.head(50), use_container_width=True)

# ============================================================================================
# GRAF 1
# ============================================================================================

st.header("Graf 1 – Qual & RTT around HO (overlaid)")

with st.expander("Editácia grafu 1", expanded=True):
    g1_title = st.text_input(
        "Názov grafu",
        value="Qual & RTT around HO (overlaid)",
        key="g1_title",
    )
    g1_xlabel = st.text_input(
        "Názov osi X",
        value="offset (samples)",
        key="g1_xlabel",
    )
    g1_ylabel_left = st.text_input(
        "Názov osi Y (ľavá)",
        value="Qual",
        key="g1_ylabel_left",
    )
    g1_ylabel_right = st.text_input(
        "Názov osi Y (pravá)",
        value="RTT_ms",
        key="g1_ylabel_right",
    )

    g1_legend_qual = st.text_input(
        "Legenda – Qual",
        value="Qual (mean)",
        key="g1_legend_qual",
    )
    g1_legend_rtt = st.text_input(
        "Legenda – RTT",
        value="RTT (mean)",
        key="g1_legend_rtt",
    )

    c1, c2 = st.columns(2)
    with c1:
        g1_color_qual = st.color_picker(
            "Farba stĺpcov – Qual",
            value="#1f77b4",
            key="g1_color_qual",
        )
    with c2:
        g1_color_rtt = st.color_picker(
            "Farba stĺpcov – RTT",
            value="#d62728",
            key="g1_color_rtt",
        )

if st.button("Vykresliť graf 1", key="btn_g1"):
    try:
        fig, _, _ = plot_qual_and_rtt_around_handovers_overlaid(
            df,
            title_text=g1_title,
            x_label=g1_xlabel,
            y_label_left=g1_ylabel_left,
            y_label_right=g1_ylabel_right,
            legend_qual=g1_legend_qual,
            legend_rtt=g1_legend_rtt,
            color_qual=g1_color_qual,
            color_rtt=g1_color_rtt,
        )
        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_1_qual_rtt_overlaid", key_prefix="g1")
    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 1: {e}")

st.divider()

# ============================================================================================
# GRAF 2
# ============================================================================================

st.header("Graf 2 – RTT around HO vs CLEAN control")

with st.expander("Editácia grafu 2", expanded=False):
    g2_title = st.text_input(
        "Názov grafu",
        value="RTT around HO vs CLEAN control (IQR)",
        key="g2_title",
    )
    g2_xlabel = st.text_input(
        "Názov osi X",
        value="offset (samples)",
        key="g2_xlabel",
    )
    g2_ylabel = st.text_input(
        "Názov osi Y",
        value="RTT_ms",
        key="g2_ylabel",
    )

    g2_use_median = st.checkbox(
        "Použiť median namiesto mean",
        value=False,
        key="g2_use_median",
    )

    g2_legend_ho_center = st.text_input(
        "Legenda – HO center",
        value="HO mean",
        key="g2_legend_ho_center",
    )
    g2_legend_ctrl_center = st.text_input(
        "Legenda – CTRL center",
        value="CTRL mean",
        key="g2_legend_ctrl_center",
    )
    g2_legend_ho_iqr = st.text_input(
        "Legenda – HO IQR",
        value="HO IQR",
        key="g2_legend_ho_iqr",
    )
    g2_legend_ctrl_iqr = st.text_input(
        "Legenda – CTRL IQR",
        value="CTRL IQR",
        key="g2_legend_ctrl_iqr",
    )

    c1, c2 = st.columns(2)
    with c1:
        g2_color_ho = st.color_picker(
            "Farba – HO",
            value="#1f77b4",
            key="g2_color_ho",
        )
    with c2:
        g2_color_ctrl = st.color_picker(
            "Farba – CTRL",
            value="#d62728",
            key="g2_color_ctrl",
        )

if st.button("Vykresliť graf 2", key="btn_g2"):
    try:
        curve = compute_rtt_around_ho_curves_clean_control(df)

        fig, _ = plot_rtt_around_ho_with_iqr_by_direction(
            curve,
            use_median=g2_use_median,
            title_text=g2_title,
            x_label=g2_xlabel,
            y_label=g2_ylabel,
            legend_ho_center=g2_legend_ho_center,
            legend_ctrl_center=g2_legend_ctrl_center,
            legend_ho_iqr=g2_legend_ho_iqr,
            legend_ctrl_iqr=g2_legend_ctrl_iqr,
            color_ho=g2_color_ho,
            color_ctrl=g2_color_ctrl,
        )
        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_2_rtt_around_ho", key_prefix="g2")

    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 2: {e}")

st.divider()

# ============================================================================================
# GRAF 3 – Heatmapa podielu RTT outlierov podľa drive_id a bucketu
# ============================================================================================

st.header("Graf 3 – Heatmapa RTT outlierov (drive_id × bucket)")

with st.expander("Editácia grafu 3", expanded=False):
    g3_title = st.text_input(
        "Názov grafu",
        value="Podiel RTT outlierov podľa drive_id a bucketu",
        key="g3_title",
    )

    g3_value_col = st.text_input(
        "RTT stĺpec", value="RTT_ms", key="g3_value_col")
    g3_drive_col = st.text_input(
        "drive_id stĺpec", value="drive_id", key="g3_drive_col")
    g3_bucket_idx_col = st.text_input(
        "bucket index stĺpec", value="common_bucket_index", key="g3_bucket_idx_col")

    g3_q1 = st.number_input("Q1 quantile", value=0.10,
                            min_value=0.0, max_value=0.49, step=0.01, key="g3_q1")
    g3_q3 = st.number_input("Q3 quantile", value=0.90,
                            min_value=0.51, max_value=1.0, step=0.01, key="g3_q3")
    g3_iqr_factor = st.number_input(
        "IQR faktor", value=1.5, min_value=0.1, max_value=10.0, step=0.1, key="g3_iqr_factor")

    g3_xlabel = st.text_input(
        "Názov osi X", value="common_bucket_index", key="g3_xlabel")
    g3_ylabel = st.text_input("Názov osi Y", value="drive_id", key="g3_ylabel")
    g3_cbar_label = st.text_input(
        "Colorbar label", value="ratio_outliers", key="g3_cbar_label")

    g3_cmap = st.selectbox(
        "Farebná škála (colormap)",
        options=[
            "viridis", "plasma", "inferno", "magma", "cividis",
            "Greys", "Reds", "Blues", "Oranges", "Purples",
            "YlOrRd", "YlGnBu", "coolwarm", "seismic",
            "turbo", "jet",
        ],
        index=0,
        key="g3_cmap",
    )

    c1, c2 = st.columns(2)
    with c1:
        g3_fig_w = st.number_input(
            "Figure width", value=10, min_value=4, max_value=40, step=1, key="g3_fig_w")
    with c2:
        g3_fig_h = st.number_input(
            "Figure height", value=6, min_value=3, max_value=30, step=1, key="g3_fig_h")

if st.button("Vykresliť graf 3", key="btn_g3"):
    try:
        df_out = filter_rtt_outliers_iqr(
            df,
            value_col=g3_value_col,
            q1=g3_q1,
            q3=g3_q3,
            iqr_factor=g3_iqr_factor,

        )

        # (voliteľné info)
        lower = df_out.attrs.get("rtt_iqr_lower", None)
        upper = df_out.attrs.get("rtt_iqr_upper", None)
        if lower is not None and upper is not None:
            st.caption(
                f"IQR hranice pre outliery: lower={lower:.2f}, upper={upper:.2f} | outliers={len(df_out):,}")

        fig, _, _ = plot_drive_bucket_outlier_heatmap(
            df_all=df,
            df_outliers=df_out,
            drive_id_col=g3_drive_col,
            bucket_index_col=g3_bucket_idx_col,
            figsize=(int(g3_fig_w), int(g3_fig_h)),
            title=g3_title,
            x_label=g3_xlabel,
            y_label=g3_ylabel,
            colorbar_label=g3_cbar_label,
            cmap=g3_cmap,
        )
        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_3_rtt_outliers_heatmap", key_prefix="g3")

    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 3: {e}")

st.divider()

# ============================================================================================
# GRAF 4
# ============================================================================================

st.header("Graf 4 – Handover rate heatmap (drive_id × bucket)")

with st.expander("Editácia grafu 4", expanded=False):
    g4_title = st.text_input(
        "Názov grafu",
        value="Handover rate podľa drive_id a bucketu",
        key="g4_title",
    )

    g4_drive_col = st.text_input(
        "drive_id stĺpec", value="drive_id", key="g4_drive_col")
    g4_bucket_idx_col = st.text_input(
        "bucket index stĺpec", value="common_bucket_index", key="g4_bucket_idx_col")
    g4_bucket_col = st.text_input(
        "bucket name stĺpec", value="bucket_name", key="g4_bucket_col")
    g4_is_ho_col = st.text_input(
        "is_handover stĺpec", value="is_handover", key="g4_is_ho_col")
    g4_rtt_col = st.text_input("RTT stĺpec", value="RTT_ms", key="g4_rtt_col")

    g4_xlabel = st.text_input(
        "Názov osi X", value="common_bucket_index", key="g4_xlabel")
    g4_ylabel = st.text_input("Názov osi Y", value="drive_id", key="g4_ylabel")
    g4_cbar_label = st.text_input(
        "Colorbar label", value="handover_rate", key="g4_cbar_label")

    g4_cmap = st.selectbox(
        "Farebná škála (colormap)",
        options=[
            "viridis", "plasma", "inferno", "magma", "cividis",
            "Greys", "Reds", "Blues", "Oranges", "Purples",
            "YlOrRd", "YlGnBu", "coolwarm", "seismic",
            "turbo", "jet",
        ],
        index=0,
        key="g4_cmap",
    )

    c1, c2 = st.columns(2)
    with c1:
        g4_fig_w = st.number_input(
            "Figure width", value=10, min_value=4, max_value=40, step=1, key="g4_fig_w")
    with c2:
        g4_fig_h = st.number_input(
            "Figure height", value=6, min_value=3, max_value=30, step=1, key="g4_fig_h")

    # voliteľné vmin/vmax (na začiatok nechaj vypnuté)
    g4_use_custom_vmax = st.checkbox(
        "Nastaviť vlastné vmax", value=False, key="g4_use_custom_vmax")
    g4_vmax = st.number_input(
        "vmax", value=1.0, min_value=0.0, step=0.05, key="g4_vmax")

if st.button("Vykresliť graf 4", key="btn_g4"):
    try:
        stats = compute_bucket_handover_stats(
            df,
            drive_id_col=g4_drive_col,
            bucket_index_col=g4_bucket_idx_col,
            bucket_col=g4_bucket_col,
            is_ho_col=g4_is_ho_col,
            rtt_col=g4_rtt_col,
        )

        vmax_arg = float(g4_vmax) if g4_use_custom_vmax else None

        fig, _, _ = plot_drive_bucket_handover_heatmap(
            stats,
            drive_id_col=g4_drive_col,
            bucket_index_col=g4_bucket_idx_col,
            figsize=(int(g4_fig_w), int(g4_fig_h)),
            title=g4_title,
            x_label=g4_xlabel,
            y_label=g4_ylabel,
            colorbar_label=g4_cbar_label,
            cmap=g4_cmap,
            vmin=0.0,
            vmax=vmax_arg,
        )
        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_4_handover_rate_heatmap", key_prefix="g4")

    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 4: {e}")

st.divider()

# ============================================================================================
# GRAF 5
# ============================================================================================

st.header("Graf 5 – RTT vs handover_rate (binned boxplot)")

with st.expander("Editácia grafu 5", expanded=False):
    g5_title = st.text_input(
        "Názov grafu",
        value="rtt_mean vs handover_rate (binned boxplots)",
        key="g5_title",
    )

    # použijeme stats per drive/bucket z compute_bucket_handover_stats
    g5_drive_col = st.text_input(
        "drive_id stĺpec", value="drive_id", key="g5_drive_col")
    g5_bucket_idx_col = st.text_input(
        "bucket index stĺpec", value="common_bucket_index", key="g5_bucket_idx_col")
    g5_bucket_col = st.text_input(
        "bucket name stĺpec", value="bucket_name", key="g5_bucket_col")
    g5_is_ho_col = st.text_input(
        "is_handover stĺpec", value="is_handover", key="g5_is_ho_col")
    g5_rtt_col_src = st.text_input(
        "RTT stĺpec (raw)", value="RTT_ms", key="g5_rtt_col_src")

    # stĺpce ktoré idú do boxplotu (z agg výstupu)
    g5_ho_col = st.text_input("X metrika (ho_col)",
                              value="handover_rate", key="g5_ho_col")
    g5_rtt_col = st.text_input(
        "Y metrika (rtt_col)", value="rtt_mean", key="g5_rtt_col")

    g5_n_bins = st.slider("Počet binov", min_value=3,
                          max_value=15, value=6, step=1, key="g5_n_bins")

    g5_xlabel = st.text_input(
        "Názov osi X", value="handover_rate bins", key="g5_xlabel")
    g5_ylabel = st.text_input("Názov osi Y", value="rtt_mean", key="g5_ylabel")

    c1, c2 = st.columns(2)
    with c1:
        g5_fig_w = st.number_input(
            "Figure width", value=8, min_value=4, max_value=30, step=1, key="g5_fig_w")
    with c2:
        g5_fig_h = st.number_input(
            "Figure height", value=4, min_value=3, max_value=20, step=1, key="g5_fig_h")

    g5_show_fliers = st.checkbox(
        "Zobraziť outliers (fliers) v boxplote", value=True, key="g5_show_fliers")
    g5_annot_counts = st.checkbox(
        "Zobraziť počty nad boxami", value=True, key="g5_annot_counts")

    g5_box_color = st.color_picker(
        "Farba boxov", value="#d3d3d3", key="g5_box_color")
    g5_box_alpha = st.slider("Opacity boxov", min_value=0.1,
                             max_value=1.0, value=0.7, step=0.05, key="g5_box_alpha")

if st.button("Vykresliť graf 5", key="btn_g5"):
    try:
        stats = compute_bucket_handover_stats(
            df,
            drive_id_col=g5_drive_col,
            bucket_index_col=g5_bucket_idx_col,
            bucket_col=g5_bucket_col,
            is_ho_col=g5_is_ho_col,
            rtt_col=g5_rtt_col_src,
        )

        fig, _, bin_info = plot_rtt_vs_handover_binned_box(
            stats_df=stats,
            ho_col=g5_ho_col,
            rtt_col=g5_rtt_col,
            n_bins=int(g5_n_bins),
            figsize=(int(g5_fig_w), int(g5_fig_h)),
            title=g5_title,
            x_label=g5_xlabel,
            y_label=g5_ylabel,
            box_facecolor=g5_box_color,
            box_alpha=float(g5_box_alpha),
            show_fliers=g5_show_fliers,
            annotate_counts=g5_annot_counts,
        )

        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_5_rtt_vs_handover_binned_box", key_prefix="g5")

    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 5: {e}")

st.divider()

# ============================================================================================
# GRAF 6
# ============================================================================================

st.header("Graf 6 – RTT boxploty podľa smeru a bucketu")

with st.expander("Editácia grafu 6", expanded=False):
    g6_title = st.text_input(
        "Názov grafu",
        value="RTT_ms per bucket by Destination",
        key="g6_title",
    )

    g6_value_col = st.text_input(
        "RTT stĺpec", value="RTT_ms", key="g6_value_col")
    g6_direction_col = st.text_input(
        "Direction stĺpec", value="Destination", key="g6_direction_col")
    g6_bucket_col = st.text_input(
        "Bucket name stĺpec", value="bucket_name", key="g6_bucket_col")
    g6_bucket_idx_col = st.text_input(
        "Bucket index stĺpec", value="common_bucket_index", key="g6_bucket_idx_col")

    g6_xlabel = st.text_input("Názov osi X", value="bucket", key="g6_xlabel")
    g6_ylabel = st.text_input("Názov osi Y", value="RTT_ms", key="g6_ylabel")

    g6_filter_outliers = st.checkbox(
        "Filtrovať outliery (per direction×bucket)", value=False, key="g6_filter_outliers")
    g6_use_log = st.checkbox("Log škála na Y", value=False, key="g6_use_log")
    g6_sharey = st.checkbox("Zdieľať Y os medzi subplotmi",
                            value=True, key="g6_sharey")
    g6_show_fliers = st.checkbox(
        "Zobraziť fliers (outliers) v boxplote", value=True, key="g6_show_fliers")

    c1, c2, c3 = st.columns(3)
    with c1:
        g6_fig_w = st.number_input(
            "Figure width", value=14, min_value=6, max_value=40, step=1, key="g6_fig_w")
    with c2:
        g6_fig_h = st.number_input(
            "Figure height", value=5, min_value=3, max_value=25, step=1, key="g6_fig_h")
    with c3:
        g6_box_alpha = st.slider("Opacity boxov", min_value=0.1,
                                 max_value=1.0, value=0.7, step=0.05, key="g6_box_alpha")

    g6_box_color = st.color_picker(
        "Farba boxov", value="#d3d3d3", key="g6_box_color")

    st.caption(
        "Outlier filter parametre (použije sa iba ak je zapnuté filtrovanie):")
    c4, c5, c6 = st.columns(3)
    with c4:
        g6_out_q1 = st.number_input(
            "Q1 percentile", value=20.0, min_value=0.0, max_value=49.0, step=1.0, key="g6_out_q1")
    with c5:
        g6_out_q3 = st.number_input(
            "Q3 percentile", value=80.0, min_value=51.0, max_value=100.0, step=1.0, key="g6_out_q3")
    with c6:
        g6_out_iqr_factor = st.number_input(
            "IQR faktor", value=1.5, min_value=0.1, max_value=10.0, step=0.1, key="g6_out_iqr_factor")

if st.button("Vykresliť graf 6", key="btn_g6"):
    try:
        fig, _ = plot_rtt_boxplots_by_direction_and_bucket(
            df=df,
            value_col=g6_value_col,
            direction_col=g6_direction_col,
            bucket_col=g6_bucket_col,
            bucket_index_col=g6_bucket_idx_col,
            filter_outliers=g6_filter_outliers,
            use_log_scale=g6_use_log,
            figsize=(int(g6_fig_w), int(g6_fig_h)),
            sharey=g6_sharey,
            title=g6_title,
            x_label=g6_xlabel,
            y_label=g6_ylabel,
            box_facecolor=g6_box_color,
            box_alpha=float(g6_box_alpha),
            show_fliers=g6_show_fliers,
            out_q1=float(g6_out_q1),
            out_q3=float(g6_out_q3),
            out_iqr_factor=float(g6_out_iqr_factor),
        )
        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_6_rtt_boxplots_by_direction_and_bucket", key_prefix="g6")

    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 6: {e}")

st.divider()

# ============================================================================================
# GRAF 7 – Outlier cloud podľa smeru (mapa v lokálnych metroch)
# ============================================================================================

st.header("Graf 7 – Outlier cloud podľa smeru (GPS)")

with st.expander("Editácia grafu 7", expanded=False):
    # outlier filter parametre (reuse)
    g7_value_col = st.text_input(
        "RTT stĺpec (na outliery)", value="RTT_ms", key="g7_value_col")
    g7_q1 = st.number_input("Q1 quantile", value=0.10,
                            min_value=0.0, max_value=0.49, step=0.01, key="g7_q1")
    g7_q3 = st.number_input("Q3 quantile", value=0.90,
                            min_value=0.51, max_value=1.0, step=0.01, key="g7_q3")
    g7_iqr_factor = st.number_input(
        "IQR faktor", value=1.5, min_value=0.1, max_value=10.0, step=0.1, key="g7_iqr_factor")

    # map parametre
    g7_drive_col = st.text_input(
        "drive_id stĺpec", value="drive_id", key="g7_drive_col")
    g7_direction_col = st.text_input(
        "Direction stĺpec", value="Destination", key="g7_direction_col")
    g7_lat_col = st.text_input(
        "Latitude stĺpec", value="Latitude", key="g7_lat_col")
    g7_lon_col = st.text_input(
        "Longitude stĺpec", value="Longitude", key="g7_lon_col")
    g7_ts_col = st.text_input(
        "Timestamp stĺpec", value="Timestamp", key="g7_ts_col")

    g7_radius = st.number_input("Radius (m) pre hustotu", value=150.0,
                                min_value=10.0, max_value=2000.0, step=10.0, key="g7_radius")

    g7_title_prefix = st.text_input(
        "Suffix v title", value="outlier cloud", key="g7_title_prefix")
    g7_cmap = st.selectbox(
        "Farebná škála (colormap)",
        options=[
            "YlOrRd", "viridis", "plasma", "inferno", "magma", "cividis",
            "Greys", "Reds", "Blues", "coolwarm", "seismic", "turbo", "jet",
        ],
        index=0,
        key="g7_cmap",
    )

    c1, c2 = st.columns(2)
    with c1:
        g7_fig_w = st.number_input(
            "Figure width", value=18, min_value=6, max_value=60, step=1, key="g7_fig_w")
    with c2:
        g7_fig_h = st.number_input(
            "Figure height", value=7, min_value=3, max_value=30, step=1, key="g7_fig_h")

    c3, c4 = st.columns(2)
    with c3:
        g7_out_size = st.number_input(
            "Veľkosť outlier bodov", value=40, min_value=5, max_value=300, step=5, key="g7_out_size")
    with c4:
        g7_out_alpha = st.slider("Opacity outlier bodov", min_value=0.1,
                                 max_value=1.0, value=0.95, step=0.05, key="g7_out_alpha")

if st.button("Vykresliť graf 7", key="btn_g7"):
    try:
        df_out = filter_rtt_outliers_iqr(
            df,
            value_col=g7_value_col,
            q1=g7_q1,
            q3=g7_q3,
            iqr_factor=g7_iqr_factor,
        )

        st.caption(
            f"Outliers s GPS: {df_out.dropna(subset=[g7_lat_col, g7_lon_col]).shape[0]:,}")

        fig, _ = plot_rtt_outlier_cloud_by_direction(
            df_outliers=df_out,
            df_all=df,
            drive_id_col=g7_drive_col,
            direction_col=g7_direction_col,
            lat_col=g7_lat_col,
            lon_col=g7_lon_col,
            ts_col=g7_ts_col,
            radius_m=float(g7_radius),
            figsize=(int(g7_fig_w), int(g7_fig_h)),
            title_prefix=g7_title_prefix,
            cmap_name=g7_cmap,
            outlier_size=int(g7_out_size),
            outlier_alpha=float(g7_out_alpha),
        )

        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_7_rtt_outlier_cloud_by_direction", key_prefix="g7")

    except Exception as e:
        st.error(f"Chyba pri kreslení Graf 7: {e}")

st.divider()

# ============================================================================================
# Graf – CellID mixing (2 heatmapy vedľa seba)
# ============================================================================================

st.header("Graf – CellID mixing heatmaps (dominant CellID + #distinct CellIDs)")

with st.expander("Editácia grafu – CellID mixing", expanded=False):
    # --- vstupné stĺpce ---
    c1, c2, c3 = st.columns(3)
    with c1:
        g8_drive_col = st.text_input(
            "drive_id stĺpec", value="drive_id", key="g8_drive_col")
    with c2:
        g8_bucket_idx_col = st.text_input(
            "bucket index stĺpec", value="common_bucket_index", key="g8_bucket_idx_col")
    with c3:
        g8_cell_col = st.text_input(
            "CellID stĺpec", value="CellID", key="g8_cell_col")

    # --- layout ---
    c4, c5 = st.columns(2)
    with c4:
        g8_fig_w = st.number_input(
            "Figure width", value=16, min_value=6, max_value=60, step=1, key="g8_fig_w")
    with c5:
        g8_fig_h = st.number_input(
            "Figure height", value=6, min_value=3, max_value=30, step=1, key="g8_fig_h")

    g8_bad_color = st.color_picker(
        "Farba pre NaN (no data)", value="#d3d3d3", key="g8_bad_color")
    g8_plus_one = st.checkbox(
        "X labely +1 (0-based → 1..N)", value=True, key="g8_plus_one")

    st.markdown("### Ľavá heatmapa: Dominant CellID")
    l1, l2 = st.columns(2)
    with l1:
        g8_title_left = st.text_input(
            "Názov (ľavá)", value="Dominant CellID per bucket/drive", key="g8_title_left")
        g8_xlabel_left = st.text_input(
            "X label (ľavá)", value="bucket index (1..N)", key="g8_xlabel_left")
        g8_ylabel_left = st.text_input(
            "Y label (ľavá)", value="drive_id", key="g8_ylabel_left")
        g8_cbar_left = st.text_input(
            "Colorbar label (ľavá)", value="dominant CellID", key="g8_cbar_left")
    with l2:
        g8_cmap_left = st.selectbox(
            "Colormap (ľavá)",
            options=["plasma", "viridis", "inferno",
                     "magma", "cividis", "turbo", "jet"],
            index=0,
            key="g8_cmap_left",
        )

    st.markdown("### Pravá heatmapa: Počet unikátnych CellID")
    r1, r2 = st.columns(2)
    with r1:
        g8_title_right = st.text_input(
            "Názov (pravá)", value="Number of distinct CellIDs per bucket/drive", key="g8_title_right")
        g8_xlabel_right = st.text_input(
            "X label (pravá)", value="bucket index (1..N)", key="g8_xlabel_right")
        g8_ylabel_right = st.text_input(
            "Y label (pravá)", value="drive_id", key="g8_ylabel_right")
        g8_cbar_right = st.text_input(
            "Colorbar label (pravá)", value="num_cellids", key="g8_cbar_right")
    with r2:
        g8_cmap_right = st.selectbox(
            "Colormap (pravá)",
            options=["Blues", "Greens", "Oranges", "Purples",
                     "Reds", "viridis", "cividis", "turbo", "jet"],
            index=0,
            key="g8_cmap_right",
        )

if st.button("Vykresliť graf – CellID mixing", key="btn_g8"):
    try:
        _, dom_pivot, cnt_pivot = compute_cellid_mixing_for_heatmaps(
            df,
            drive_id_col=g8_drive_col,
            bucket_index_col=g8_bucket_idx_col,
            cell_col=g8_cell_col,
        )

        fig, _ = plot_cellid_mixing_heatmaps(
            dominant_pivot=dom_pivot,
            num_cellids_pivot=cnt_pivot,
            figsize=(int(g8_fig_w), int(g8_fig_h)),
            # left
            title_left=g8_title_left,
            xlabel_left=g8_xlabel_left,
            ylabel_left=g8_ylabel_left,
            cmap_left=g8_cmap_left,
            cbar_label_left=g8_cbar_left,
            # right
            title_right=g8_title_right,
            xlabel_right=g8_xlabel_right,
            ylabel_right=g8_ylabel_right,
            cmap_right=g8_cmap_right,
            cbar_label_right=g8_cbar_right,
            # misc
            bad_color=g8_bad_color,
            x_labels_plus_one=g8_plus_one,
        )

        st.pyplot(fig, use_container_width=True)
        _export_figure_buttons(
            fig, base_name="graf_8_cellid_mixing_heatmaps", key_prefix="g8")

    except Exception as e:
        st.error(f"Chyba pri kreslení CellID mixing heatmaps: {e}")

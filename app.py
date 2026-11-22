from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Iowa Nitrate Monitoring & Predictive Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(15, 76, 129, 0.04);
        border: 1px solid rgba(15, 76, 129, 0.16);
        border-radius: 10px;
        padding: 14px 18px;
    }
    .insight-box {
        border-left: 4px solid #0D9488;
        background-color: rgba(13, 148, 136, 0.07);
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 0.75rem 0 1.25rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(".")
RESULTS_DIR = DATA_DIR / "Iowa_Nitrate_All_Methods_Results"

PALETTE = {
    "no_manure": "#1E3A5F",
    "with_manure": "#0D9488",
    "baseline": "#94A3B8",
    "overpred": "#D97706",
    "underpred": "#0284C7",
}

METHOD_ORDER = ["A_original_all_available", "B_matched_station_years", "C_balanced_panel"]
GRAIN_ORDER = ["daily", "weekly", "monthly", "season_year"]

SHORT_LABELS = {
    "A_original_all_available": "Original (Complete Dataset)",
    "B_matched_station_years": "New (Matched Dataset)",
    "C_balanced_panel": "Modified (Balanced Panel)",
}

DESIGN_SUBTITLES = {
    "A_original_all_available": "Original (Complete Dataset) — All Available Monitoring Station-Years",
    "B_matched_station_years": "New (Matched Dataset) — Qualifying Station-Years (≥80% Annual Coverage)",
    "C_balanced_panel": "Modified (Balanced Panel) — Fixed Cohort of Top 7 Stations × 7 Years",
}

METHOD_COLORS = {
    "Original (Complete Dataset)": "#64748B",
    "New (Matched Dataset)": "#1E3A5F",
    "Modified (Balanced Panel)": "#0D9488",
}


@st.cache_data
def load_master_tables():
    out = {}
    files = {
        "cv_all": "all_methods_spatial_cv_metrics.csv",
        "ho_all": "all_methods_temporal_holdout_metrics.csv",
        "fi_all": "all_methods_feature_importances.csv",
        "manure_av": "all_methods_manure_added_value.csv",
    }
    for key, fn in files.items():
        p = RESULTS_DIR / fn
        out[key] = pd.read_csv(p) if p.is_file() else pd.DataFrame()
    return out


@st.cache_data
def load_slice_files(design_option: str, timescale: str):
    base = RESULTS_DIR / design_option / timescale
    res = {}
    for name in ["spatial_cv_metrics.csv", "temporal_holdout_metrics.csv", "station_bias_summary.csv", "oof_predictions_sample.csv"]:
        p = base / name
        res[name] = pd.read_csv(p) if p.is_file() else pd.DataFrame()
    return res


tables = load_master_tables()

with st.sidebar:
    st.header("Analysis Controls")
    available_designs = [d for d in METHOD_ORDER if (RESULTS_DIR / d).is_dir()] or METHOD_ORDER
    design_option = st.selectbox(
        "Dataset Cohort",
        options=available_designs,
        format_func=lambda x: SHORT_LABELS.get(x, x),
    )

    design_path = RESULTS_DIR / design_option
    if design_path.is_dir():
        found = [p.name for p in design_path.iterdir() if p.is_dir()]
        timescales = [t for t in GRAIN_ORDER if t in found] + [t for t in sorted(found) if t not in GRAIN_ORDER]
    else:
        timescales = GRAIN_ORDER

    timescale = st.selectbox(
        "Temporal Resolution",
        options=timescales,
        format_func=lambda x: x.replace("_", "-").title(),
    )

    st.divider()
    st.markdown(
        """
        **Cohort Definitions**
        - **Original (Complete Dataset):** All available station-years with no completeness filtering.
        - **New (Matched Dataset):** Station-years meeting ≥80% March–November observation coverage.
        - **Modified (Balanced Panel):** Constant 7-station × 7-year matrix removing site-composition drift.
        """
    )

slice_data = load_slice_files(design_option, timescale)
df_cv = slice_data["spatial_cv_metrics.csv"]

st.title("Iowa Nitrate Monitoring & Predictive Dashboard")
st.caption(DESIGN_SUBTITLES.get(design_option, design_option))

if df_cv.empty:
    st.warning("Results not found for the selected configuration.")
    st.stop()

ml_rows = df_cv[df_cv["model"] != "Mean-only"].copy()
baseline_row = df_cv[df_cv["model"] == "Mean-only"]
baseline_rmse = float(baseline_row["RMSE"].iloc[0]) if not baseline_row.empty else np.nan

best_row = ml_rows.loc[ml_rows["RMSE"].idxmin()]
best_model_name = best_row["model"]
same_model = ml_rows[ml_rows["model"] == best_model_name]

no_man_rmse = same_model.loc[same_model["feature_set"] == "Without manure", "RMSE"]
with_man_rmse = same_model.loc[same_model["feature_set"] == "With manure", "RMSE"]
manure_delta = (
    float(with_man_rmse.iloc[0] - no_man_rmse.iloc[0])
    if (not no_man_rmse.empty and not with_man_rmse.empty)
    else None
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        label=f"Lowest Spatial RMSE ({best_model_name})",
        value=f"{best_row['RMSE']:.2f} mg/L",
        delta=f"{manure_delta:+.2f} mg/L vs. no manure" if manure_delta is not None else None,
        delta_color="inverse",
    )
with k2:
    nse_delta = (
        f"{best_row['NSE'] - float(baseline_row['NSE'].iloc[0]):+.2f} vs. mean baseline"
        if not baseline_row.empty
        else None
    )
    st.metric(label="Nash-Sutcliffe Efficiency (NSE)", value=f"{best_row['NSE']:.2f}", delta=nse_delta)
with k3:
    st.metric(
        label="Percent Bias (PBIAS)",
        value=f"{best_row['PBIAS_pct']:+.1f}%",
        delta="Within ±10% target" if abs(best_row["PBIAS_pct"]) <= 10 else "Exceeds ±10%",
        delta_color="normal" if abs(best_row["PBIAS_pct"]) <= 10 else "inverse",
    )
with k4:
    st.metric(
        label="Cohort Sample Size",
        value=f"{int(best_row['n']):,} obs",
        delta=f"{int(best_row['n_stations'])} stations · {int(best_row['n_groups'])} HUC-8s",
        delta_color="off",
    )

if manure_delta is not None:
    pct_change = (manure_delta / float(no_man_rmse.iloc[0])) * 100
    direction = "reduces" if manure_delta < 0 else "increases"
    st.markdown(
        f"""
        <div class="insight-box">
        <strong>Key Finding ({SHORT_LABELS.get(design_option, design_option)} · {timescale.replace('_', '-').title()}):</strong>
        Adding manure predictors <strong>{direction} {best_model_name} RMSE by {abs(manure_delta):.2f} mg/L ({abs(pct_change):.1f}%)</strong>
        (from {float(no_man_rmse.iloc[0]):.2f} to {float(with_man_rmse.iloc[0]):.2f} mg/L as N) with an MAE of <strong>{best_row['MAE']:.2f} mg/L</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_models, tab_compare, tab_scales, tab_spatial = st.tabs(
    [
        "📊 Model & Manure Comparison",
        "⚖️ Original vs. New vs. Modified",
        "📈 Timescale & Holdout Scaling",
        "🗺️ Station Bias, Fit & Features",
    ]
)

with tab_models:
    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(
            ml_rows, x="model", y="RMSE", color="feature_set", barmode="group",
            text=ml_rows["RMSE"].map(lambda v: f"{v:.2f}"),
            color_discrete_map={"Without manure": PALETTE["no_manure"], "With manure": PALETTE["with_manure"]},
            labels={"model": "", "RMSE": "Spatial CV RMSE (mg/L as N)", "feature_set": "Predictor Set"},
            title="Manure Predictor Impact on Spatial RMSE (Lower is Better)",
        )
        if not np.isnan(baseline_rmse):
            fig_rmse.add_hline(y=baseline_rmse, line_dash="dash", line_color=PALETTE["baseline"], annotation_text=f"Mean baseline ({baseline_rmse:.2f})")
        fig_rmse.update_traces(textposition="outside", cliponaxis=False)
        fig_rmse.update_layout(height=410, legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c2:
        fig_nse = px.bar(
            ml_rows, x="model", y="NSE", color="feature_set", barmode="group",
            text=ml_rows["NSE"].map(lambda v: f"{v:.2f}"),
            color_discrete_map={"Without manure": PALETTE["no_manure"], "With manure": PALETTE["with_manure"]},
            labels={"model": "", "NSE": "Nash-Sutcliffe Efficiency (NSE)", "feature_set": "Predictor Set"},
            title="Model Explanatory Skill (Higher is Better)",
        )
        fig_nse.add_hline(y=0, line_dash="dot", line_color=PALETTE["baseline"])
        fig_nse.update_traces(textposition="outside", cliponaxis=False)
        fig_nse.update_layout(height=410, legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
        st.plotly_chart(fig_nse, use_container_width=True)

    with st.expander("View Formatted Metrics Table & Download CSV"):
        cols = ["model", "feature_set", "RMSE", "station_equal_RMSE", "MAE", "NSE", "R2", "PBIAS_pct", "n"]
        avail = [c for c in cols if c in df_cv.columns]
        clean_df = df_cv[avail].rename(
            columns={
                "model": "Model", "feature_set": "Feature Set", "RMSE": "RMSE (mg/L)",
                "station_equal_RMSE": "Station-Equal RMSE", "MAE": "MAE (mg/L)",
                "NSE": "NSE", "R2": "R²", "PBIAS_pct": "PBIAS (%)", "n": "Observations (n)",
            }
        )
        st.dataframe(
            clean_df.style.format(
                {
                    "RMSE (mg/L)": "{:.3f}", "Station-Equal RMSE": "{:.3f}", "MAE (mg/L)": "{:.3f}",
                    "NSE": "{:.3f}", "R²": "{:.3f}", "PBIAS (%)": "{:+.2f}%", "Observations (n)": "{:,}",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button("Download Metrics CSV", data=clean_df.to_csv(index=False), file_name=f"{design_option}_{timescale}_metrics.csv", mime="text/csv")

with tab_compare:
    cv_all = tables["cv_all"]
    man_av = tables["manure_av"]
    if not cv_all.empty:
        sub_grain = cv_all[(cv_all["grain"] == timescale) & (cv_all["model"] != "Mean-only")].copy()
        sub_grain["Cohort"] = sub_grain["panel_design"].map(lambda x: SHORT_LABELS.get(x, x))
        ordered_labels = ["Original (Complete Dataset)", "New (Matched Dataset)", "Modified (Balanced Panel)"]

        c_left, c_right = st.columns(2)
        with c_left:
            with_m = sub_grain[sub_grain["feature_set"] == "With manure"]
            fig_m_rmse = px.bar(
                with_m, x="model", y="RMSE", color="Cohort", barmode="group",
                text=with_m["RMSE"].map(lambda v: f"{v:.2f}"),
                category_orders={"Cohort": ordered_labels}, color_discrete_map=METHOD_COLORS,
                labels={"model": "", "RMSE": "Spatial CV RMSE (With Manure, mg/L)"},
                title=f"Cohort Spatial Error Comparison ({timescale.replace('_', '-').title()})",
            )
            fig_m_rmse.update_traces(textposition="outside", cliponaxis=False)
            fig_m_rmse.update_layout(height=420, legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
            st.plotly_chart(fig_m_rmse, use_container_width=True)

        with c_right:
            if not man_av.empty and "RMSE_delta" in man_av.columns:
                av_sub = man_av[man_av["grain"] == timescale].copy()
                av_sub["Cohort"] = av_sub["panel_design"].map(lambda x: SHORT_LABELS.get(x, x))
                fig_av = px.bar(
                    av_sub, x="model", y="RMSE_delta", color="Cohort", barmode="group",
                    text=av_sub["RMSE_delta"].map(lambda v: f"{v:+.2f}"),
                    category_orders={"Cohort": ordered_labels}, color_discrete_map=METHOD_COLORS,
                    labels={"model": "", "RMSE_delta": "Manure ΔRMSE (mg/L, Negative = Improvement)"},
                    title=f"Manure Added-Value Across Cohorts ({timescale.replace('_', '-').title()})",
                )
                fig_av.add_hline(y=0, line_dash="dash", line_color="#64748B")
                fig_av.update_traces(textposition="outside", cliponaxis=False)
                fig_av.update_layout(height=420, legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
                st.plotly_chart(fig_av, use_container_width=True)

with tab_scales:
    cv_all = tables["cv_all"]
    ho_df = slice_data["temporal_holdout_metrics.csv"]
    c1, c2 = st.columns(2)
    with c1:
        if not cv_all.empty:
            sub_sc = cv_all[(cv_all["panel_design"] == design_option) & (cv_all["model"] != "Mean-only")].copy()
            sub_sc["grain"] = pd.Categorical(sub_sc["grain"], categories=GRAIN_ORDER, ordered=True)
            sub_sc = sub_sc.sort_values("grain")
            fig_sc = px.line(
                sub_sc, x="grain", y="RMSE", color="model", line_dash="feature_set", markers=True,
                labels={"grain": "Temporal Resolution", "RMSE": "Spatial CV RMSE (mg/L)"},
                title="Error Reduction Across Temporal Aggregation Scales",
            )
            fig_sc.update_layout(height=410)
            st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        if not ho_df.empty:
            ho_ml = ho_df[ho_df["model"] != "Mean-only"].copy()
            fig_ho = px.bar(
                ho_ml, x="model", y="RMSE", color="feature_set", barmode="group",
                text=ho_ml["RMSE"].map(lambda v: f"{v:.2f}"),
                color_discrete_map={"Without manure": PALETTE["no_manure"], "With manure": PALETTE["with_manure"]},
                labels={"model": "", "RMSE": "2023–2025 Holdout RMSE (mg/L)"},
                title=f"Temporal Holdout Validation ({timescale.replace('_', '-').title()})",
            )
            fig_ho.update_traces(textposition="outside", cliponaxis=False)
            fig_ho.update_layout(height=410, legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
            st.plotly_chart(fig_ho, use_container_width=True)
        else:
            st.info("Temporal holdout metrics not available for this slice.")

with tab_spatial:
    bias_df = slice_data["station_bias_summary.csv"]
    preds_df = slice_data["oof_predictions_sample.csv"]
    fi_all = tables["fi_all"]

    model_choices = [m for m in ml_rows["model"].unique()]
    sel_model = st.selectbox(
        "Inspect Model Diagnostics",
        options=model_choices,
        index=int(np.argmax(np.array(model_choices) == best_model_name)),
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if not bias_df.empty:
            b_sub = bias_df[(bias_df["model"] == sel_model) & (bias_df["feature_set"] == "With manure")].sort_values("mean_bias")
            b_sub["Direction"] = np.where(b_sub["mean_bias"] >= 0, "Underpredicted (Obs > Pred)", "Overpredicted (Pred > Obs)")
            fig_bias = px.bar(
                b_sub, x="mean_bias", y="station_id", orientation="h", color="Direction",
                color_discrete_map={"Underpredicted (Obs > Pred)": PALETTE["overpred"], "Overpredicted (Pred > Obs)": PALETTE["underpred"]},
                labels={"mean_bias": "Mean Bias: Observed − Predicted (mg/L)", "station_id": "USGS Station"},
                title=f"Station-Level Bias ({sel_model}, With Manure)",
            )
            fig_bias.add_vline(x=0, line_dash="dash", line_color="#64748B")
            fig_bias.update_layout(height=420, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
            st.plotly_chart(fig_bias, use_container_width=True)

    with col_b:
        if not preds_df.empty:
            p_sub = preds_df[(preds_df["model"] == sel_model) & (preds_df["feature_set"] == "With manure")]
            if not p_sub.empty:
                lim_max = float(max(p_sub["observed"].max(), p_sub["predicted"].max())) * 1.05
                fig_scat = px.scatter(
                    p_sub, x="observed", y="predicted", color="station_id", opacity=0.65,
                    labels={"observed": "Observed Nitrate-N (mg/L)", "predicted": "Predicted Nitrate-N (mg/L)"},
                    title=f"Observed vs. Predicted ({sel_model}, Spatial CV)",
                )
                fig_scat.add_shape(type="line", x0=0, y0=0, x1=lim_max, y1=lim_max, line=dict(color="#64748B", dash="dash"))
                fig_scat.update_layout(height=420, showlegend=False)
                st.plotly_chart(fig_scat, use_container_width=True)

    if not fi_all.empty:
        fi_sub = fi_all[
            (fi_all["panel_design"] == design_option) & (fi_all["grain"] == timescale) & (fi_all["model"] == sel_model)
        ].sort_values("importance", ascending=True).tail(12)
        if not fi_sub.empty:
            fi_sub["Predictor Type"] = np.where(fi_sub["is_manure"], "Manure Predictor", "Base Spatial / Seasonal")
            fig_fi = px.bar(
                fi_sub, x="importance", y="feature", orientation="h", color="Predictor Type",
                color_discrete_map={"Manure Predictor": PALETTE["with_manure"], "Base Spatial / Seasonal": PALETTE["no_manure"]},
                labels={"importance": "Relative Feature Importance", "feature": ""},
                title=f"Top Predictors Driving {sel_model} ({timescale.replace('_', '-').title()})",
            )
            fig_fi.update_layout(height=360, legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
            st.plotly_chart(fig_fi, use_container_width=True)

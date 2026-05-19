"""
Streamlit GUI — reads from pipeline.db and displays AI interpretations.
Run:  streamlit run app.py
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

import db

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"

def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_db_path = Path(__file__).parent / _cfg["db"]["path"]
db.init_db(_db_path)

st.set_page_config(
    page_title=_cfg["gui"]["page_title"],
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached queries ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_patient_table(
    pred_class: str,
    tier: str,
    cancer_type: str,
    top_system: str,
    sort_col: str,
    sort_asc: bool,
) -> pd.DataFrame:
    sql = """
        SELECT p.sample_id, p.patient_id, p.predicted_prob, p.predicted_class,
               p.binary_label, p.os_months, p.os_status,
               p.cancer_type_detailed, p.sample_type,
               p.mutation_count, p.tumor_purity,
               i.confidence_tier, i.top_driver_system, i.top_driver_gene,
               i.hallucination_flag, i.interpreted_at
        FROM patients p
        LEFT JOIN interpretations i ON p.sample_id = i.sample_id
        WHERE 1=1
    """
    params = []
    if pred_class != "All":
        sql += " AND p.predicted_class = ?"
        params.append(int(pred_class))
    if tier != "All":
        sql += " AND i.confidence_tier = ?"
        params.append(tier)
    if cancer_type != "All":
        sql += " AND p.cancer_type_detailed = ?"
        params.append(cancer_type)
    if top_system != "All":
        sql += " AND i.top_driver_system = ?"
        params.append(top_system)
    order = "ASC" if sort_asc else "DESC"
    sql += f" ORDER BY p.{sort_col} {order}"

    with db.connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


@st.cache_data(ttl=300)
def load_patient_detail(sample_id: str) -> dict:
    with db.connect() as conn:
        patient = dict(conn.execute(
            "SELECT * FROM patients WHERE sample_id = ?", (sample_id,)
        ).fetchone() or {})

        genes = pd.read_sql_query(
            """SELECT gi.gene, gi.importance, gi.signed_z, cg.rho AS cohort_rho
               FROM patient_gene_importance gi
               LEFT JOIN cohort_gene_scores cg ON gi.gene = cg.gene
               WHERE gi.sample_id = ?
               ORDER BY gi.importance DESC LIMIT 20""",
            conn, params=(sample_id,),
        )

        systems = pd.read_sql_query(
            """SELECT pr.term_id, pr.rlipp_score AS patient_rlipp,
                      cr.rlipp AS cohort_rlipp, cr.p_rho AS cohort_p_rho
               FROM patient_rlipp pr
               LEFT JOIN cohort_rlipp cr ON pr.term_id = cr.term_id
               WHERE pr.sample_id = ?
               ORDER BY pr.rlipp_score DESC LIMIT 10""",
            conn, params=(sample_id,),
        )

        interp_row = conn.execute(
            "SELECT * FROM interpretations WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        interp = dict(interp_row) if interp_row else None
        if interp and interp.get("interpretation_json"):
            interp["parsed"] = json.loads(interp["interpretation_json"])

    return {"patient": patient, "genes": genes, "systems": systems, "interp": interp}


@st.cache_data(ttl=3600)
def load_cohort_summary() -> dict:
    with db.connect() as conn:
        top_systems = pd.read_sql_query(
            "SELECT term_id, rlipp, p_rho FROM cohort_rlipp ORDER BY rlipp DESC LIMIT 15", conn
        )
        top_genes = pd.read_sql_query(
            "SELECT gene, rho FROM cohort_gene_scores ORDER BY ABS(rho) DESC LIMIT 20", conn
        )
        prob_hist = pd.read_sql_query(
            "SELECT predicted_prob, binary_label FROM patients", conn
        )
        counts = pd.read_sql_query(
            """SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN i.sample_id IS NOT NULL THEN 1 ELSE 0 END) AS interpreted,
                SUM(CASE WHEN i.confidence_tier='high' THEN 1 ELSE 0 END) AS high,
                SUM(CASE WHEN i.confidence_tier='moderate' THEN 1 ELSE 0 END) AS moderate,
                SUM(CASE WHEN i.confidence_tier='low' THEN 1 ELSE 0 END) AS low
               FROM patients p LEFT JOIN interpretations i ON p.sample_id = i.sample_id""",
            conn,
        )
    return {
        "top_systems": top_systems,
        "top_genes": top_genes,
        "prob_hist": prob_hist,
        "counts": counts,
    }


@st.cache_data(ttl=3600)
def load_filter_options() -> dict:
    with db.connect() as conn:
        cancer_types = [r[0] for r in conn.execute(
            "SELECT DISTINCT cancer_type_detailed FROM patients WHERE cancer_type_detailed IS NOT NULL ORDER BY 1"
        ).fetchall()]
        top_systems = [r[0] for r in conn.execute(
            "SELECT DISTINCT top_driver_system FROM interpretations WHERE top_driver_system IS NOT NULL ORDER BY 1"
        ).fetchall()]
    return {"cancer_types": cancer_types, "top_systems": top_systems}


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    opts = load_filter_options()
    st.sidebar.header("Filters")
    pred_class = st.sidebar.selectbox("Predicted class", ["All", "0 (unfavorable)", "1 (favorable)"])
    pred_class_val = "All" if pred_class == "All" else pred_class[0]

    tier = st.sidebar.selectbox("Confidence tier", ["All", "high", "moderate", "low"])
    cancer_type = st.sidebar.selectbox("Cancer type", ["All"] + opts["cancer_types"])
    top_system = st.sidebar.selectbox("Top driver system", ["All"] + opts["top_systems"])

    st.sidebar.header("Sort")
    sort_col = st.sidebar.selectbox("Sort by", ["predicted_prob", "os_months", "mutation_count"])
    sort_asc = st.sidebar.checkbox("Ascending", value=False)

    return {
        "pred_class": pred_class_val,
        "tier": tier,
        "cancer_type": cancer_type,
        "top_system": top_system,
        "sort_col": sort_col,
        "sort_asc": sort_asc,
    }


# ── Tab renderers ─────────────────────────────────────────────────────────────

def render_patient_table(df: pd.DataFrame) -> None:
    st.subheader(f"Patients ({len(df):,})")

    if df.empty:
        st.info("No patients match the current filters.")
        return

    display = df[[
        "sample_id", "predicted_prob", "predicted_class", "binary_label",
        "os_months", "os_status", "cancer_type_detailed",
        "confidence_tier", "top_driver_system", "top_driver_gene",
        "hallucination_flag", "interpreted_at",
    ]].copy()

    display["predicted_prob"] = display["predicted_prob"].round(4)
    display["os_months"] = display["os_months"].round(1)

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "predicted_prob": st.column_config.ProgressColumn(
                "Pred. prob", min_value=0, max_value=1, format="%.3f"
            ),
            "hallucination_flag": st.column_config.CheckboxColumn("⚠ Flag"),
        },
    )

    selected = event.selection.rows if hasattr(event, "selection") else []
    if selected:
        sid = display.iloc[selected[0]]["sample_id"]
        st.session_state["selected_sample_id"] = sid
        st.success(f"Selected: {sid} — switch to Patient Detail tab.")


def render_patient_detail(sample_id: str) -> None:
    detail = load_patient_detail(sample_id)
    p = detail["patient"]
    genes_df = detail["genes"]
    systems_df = detail["systems"]
    interp = detail["interp"]

    st.subheader(f"Patient: {sample_id}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Clinical & Prediction**")
        info = {
            "OS (months)": p.get("os_months"),
            "OS status": p.get("os_status"),
            "Predicted prob": f"{p.get('predicted_prob', 0):.4f}",
            "Predicted class": p.get("predicted_class"),
            "True label": p.get("binary_label"),
            "Cancer type": p.get("cancer_type_detailed"),
            "Sample type": p.get("sample_type"),
            "Mutation count": p.get("mutation_count"),
            "Tumor purity": p.get("tumor_purity"),
            "Gene panel": p.get("gene_panel"),
        }
        for k, v in info.items():
            if v is not None:
                st.markdown(f"**{k}:** {v}")

    with col2:
        if not genes_df.empty:
            genes_df["color"] = genes_df["signed_z"].apply(
                lambda z: "positive" if (z or 0) >= 0 else "negative"
            )
            fig = px.bar(
                genes_df.head(15), x="importance", y="gene",
                orientation="h", color="color",
                color_discrete_map={"positive": "#e07b54", "negative": "#5b8db8"},
                title="Top genes by importance",
                labels={"importance": "|z-score|", "gene": ""},
            )
            fig.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if not systems_df.empty:
            fig2 = px.bar(
                systems_df.head(10), x="patient_rlipp", y="term_id",
                orientation="h", color="cohort_rlipp",
                color_continuous_scale="RdYlGn",
                title="Top systems by patient RLIPP",
                labels={"patient_rlipp": "Patient RLIPP", "term_id": ""},
            )
            fig2.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0))
            fig2.update_yaxes(autorange="reversed")
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    if interp is None:
        st.warning("No AI interpretation yet. Run `python interpret.py` to generate one.")
        return

    if interp.get("hallucination_flag"):
        st.warning("⚠ This interpretation has been flagged — the AI may have cited an inaccurate probability. Review carefully.")

    parsed = interp.get("parsed", {})
    if not parsed:
        st.error("Could not parse stored interpretation JSON.")
        return

    st.markdown(f"**Interpreted at:** {interp.get('interpreted_at')} &nbsp;|&nbsp; "
                f"**Model:** {interp.get('ai_model')} &nbsp;|&nbsp; "
                f"**Tokens:** {interp.get('tokens_used')}")
    st.divider()

    for field_key, field_value in parsed.items():
        label = field_key.replace("_", " ").title()
        with st.expander(f"📋 {label}", expanded=(field_key == "survival_prediction_summary")):
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, dict):
                        st.json(item)
                    else:
                        st.markdown(f"- {item}")
            elif isinstance(field_value, dict):
                st.json(field_value)
            elif field_value is None:
                st.markdown("*null*")
            else:
                st.markdown(str(field_value))

    with st.expander("🔢 Raw DB1 numbers (audit)"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top genes**")
            st.dataframe(genes_df.head(10), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Top systems**")
            st.dataframe(systems_df.head(10), use_container_width=True, hide_index=True)


def render_cohort_summary() -> None:
    data = load_cohort_summary()
    counts = data["counts"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total patients", f"{int(counts['total']):,}")
    col2.metric("Interpreted", f"{int(counts['interpreted'] or 0):,}")
    col3.metric("High confidence", f"{int(counts['high'] or 0):,}")
    col4.metric("Coverage", f"{100*(counts['interpreted'] or 0)/max(counts['total'],1):.1f}%")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            data["prob_hist"], x="predicted_prob", color="binary_label",
            nbins=40, barmode="overlay", opacity=0.7,
            title="Predicted probability distribution",
            labels={"predicted_prob": "Predicted probability", "binary_label": "True label"},
            color_discrete_map={0: "#e07b54", 1: "#5b8db8"},
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
            data["top_systems"], x="rlipp", y="term_id",
            orientation="h", color="p_rho",
            color_continuous_scale="RdYlGn",
            title="Top systems by cohort RLIPP",
            labels={"rlipp": "RLIPP", "term_id": "System", "p_rho": "p_rho"},
        )
        fig2.update_yaxes(autorange="reversed")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    genes = data["top_genes"].copy()
    genes["direction"] = genes["rho"].apply(lambda r: "positive" if r >= 0 else "negative")
    fig3 = px.bar(
        genes, x="rho", y="gene", orientation="h",
        color="direction",
        color_discrete_map={"positive": "#e07b54", "negative": "#5b8db8"},
        title="Top genes by cohort Spearman ρ",
        labels={"rho": "Spearman ρ (cohort)", "gene": ""},
    )
    fig3.update_yaxes(autorange="reversed")
    fig3.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title(_cfg["gui"]["page_title"])

    filters = render_sidebar()
    tab1, tab2, tab3 = st.tabs(["Patient Table", "Patient Detail", "Cohort Summary"])

    with tab1:
        df = load_patient_table(**filters)
        render_patient_table(df)

    with tab2:
        sid = st.session_state.get("selected_sample_id")
        if sid:
            render_patient_detail(sid)
        else:
            st.info("Select a patient from the Patient Table tab.")

    with tab3:
        render_cohort_summary()


if __name__ == "__main__":
    main()

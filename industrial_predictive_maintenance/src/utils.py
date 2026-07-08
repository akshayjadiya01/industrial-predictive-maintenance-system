from __future__ import annotations

import hmac
import os
from io import BytesIO
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def configure_page() -> None:
    """Configure the Streamlit page layout and styling."""
    st.set_page_config(page_title="Industrial Predictive Maintenance", page_icon="🏭", layout="wide")


def get_auth_credentials() -> tuple[str, str]:
    """Resolve auth credentials from Streamlit secrets or environment variables."""
    try:
        if hasattr(st, "secrets") and "app" in st.secrets:
            return str(st.secrets["app"]["username"]), str(st.secrets["app"]["password"])
    except Exception:
        pass
    return os.getenv("APP_USERNAME", "admin"), os.getenv("APP_PASSWORD", "industrial2026")


def render_authentication() -> bool:
    """Render a simple access gate for the dashboard."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.sidebar.markdown("### Secure Access")
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    submitted = st.sidebar.button("Login")

    expected_username, expected_password = get_auth_credentials()
    if submitted:
        if hmac.compare_digest(username_input, expected_username) and hmac.compare_digest(password_input, expected_password):
            st.session_state.authenticated = True
            st.sidebar.success("Authenticated")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

    if not st.session_state.authenticated:
        st.sidebar.caption("Set APP_USERNAME and APP_PASSWORD in Streamlit Cloud secrets for production access.")
        st.stop()
    return True


def apply_dashboard_theme() -> None:
    """Inject premium glassmorphism styling for the dashboard."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(7, 17, 31, 0.78);
            --panel-strong: rgba(13, 25, 44, 0.9);
            --border: rgba(148, 163, 184, 0.2);
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-2: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        .stApp {
            background: radial-gradient(circle at top left, #12304a 0%, var(--bg) 45%, #030711 100%);
            color: var(--text);
        }
        [data-testid="stSidebar"] {
            background: rgba(2, 8, 23, 0.82);
            border-right: 1px solid var(--border);
            backdrop-filter: blur(14px);
        }
        .hero-panel, .glass-panel, .kpi-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(11, 20, 36, 0.86));
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.25);
            backdrop-filter: blur(20px);
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
        }
        .kpi-card {
            animation: fadeIn 0.6s ease;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 50px rgba(56, 189, 248, 0.16);
        }
        .kpi-title {
            font-size: 0.83rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text);
        }
        .kpi-subtitle {
            font-size: 0.85rem;
            color: #7dd3fc;
            margin-top: 0.2rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            background: rgba(34, 197, 94, 0.16);
            color: #bbf7d0;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .status-pill.warning { background: rgba(245, 158, 11, 0.2); color: #fde68a; border-color: rgba(245, 158, 11, 0.3); }
        .status-pill.danger { background: rgba(239, 68, 68, 0.22); color: #fecaca; border-color: rgba(239, 68, 68, 0.3); }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, icon: str = "📈", subtitle: str = "") -> None:
    """Render a premium KPI card in the Streamlit UI."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{icon} {title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str) -> None:
    """Render a glassmorphism content panel."""
    st.markdown(
        f"""
        <div class="glass-panel">
            <h4 style="margin:0 0 0.35rem 0; color:#f8fafc;">{title}</h4>
            <div style="color:#cbd5e1;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_probability_gauge(probability: float) -> go.Figure:
    """Create a premium gauge for failure probability."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={"x": [0.15, 0.85], "y": [0.1, 0.95]},
            title={"text": "Failure Probability"},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "white"},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(34, 197, 94, 0.22)"},
                    {"range": [50, 80], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [80, 100], "color": "rgba(239, 68, 68, 0.24)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_confusion_matrix_plot(confusion_matrix: Any) -> go.Figure:
    """Create an interactive confusion matrix figure."""
    labels = ["Healthy", "Failure"]
    fig = go.Figure(data=go.Heatmap(z=confusion_matrix, x=labels, y=labels, colorscale="Bluered"))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_prediction_report_pdf(payload: dict[str, Any], probability: float, status: str, recommendation: str) -> bytes:
    """Generate a downloadable PDF report for a prediction."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Industrial Predictive Maintenance Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Status: {status}", styles["Heading2"]))
    story.append(Paragraph(f"Failure Probability: {probability * 100:.1f}%", styles["Heading3"]))
    story.append(Paragraph(f"Recommendation: {recommendation}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    rows = [["Parameter", "Value"]]
    for key, value in payload.items():
        rows.append([key, str(value)])
    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    story.append(table)
    doc.build(story)
    return buffer.getvalue()


def get_feature_importance_table(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importance if available."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = [0.0] * len(feature_names)
    return pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)

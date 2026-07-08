from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.data_preprocessing import load_dataset, prepare_training_dataframe
from src.train_model import train_and_save_model
from src.predict import load_model_bundle, predict_failure, get_maintenance_recommendation, calculate_health_score
from src.utils import (
    apply_dashboard_theme,
    build_confusion_matrix_plot,
    build_prediction_report_pdf,
    build_probability_gauge,
    configure_page,
    render_authentication,
    render_metric_card,
    render_panel,
)

configure_page()
apply_dashboard_theme()
render_authentication()

DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"


def build_fallback_artifact() -> dict:
    """Create a safe fallback artifact when model training is unavailable."""
    health_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=72,
            domain={"x": [0.15, 0.85], "y": [0.1, 0.95]},
            title={"text": "Machine Health"},
            gauge={"axis": {"range": [None, 100]}, "bar": {"color": "#38bdf8"}},
        )
    )
    return {
        "model_name": "Demo Mode",
        "feature_names": [
            "Air Temperature",
            "Process Temperature",
            "Rotational Speed",
            "Torque",
            "Tool Wear",
            "Product Type_L",
            "Product Type_M",
            "Product Type_H",
        ],
        "metrics": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        },
        "comparison_table": pd.DataFrame([{"Model": "Demo", "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1 Score": 0.0, "ROC AUC": 0.0}]),
        "shap_summary": pd.DataFrame([{"Feature": "Air Temperature", "Mean Abs SHAP": 0.0}]),
        "health_gauge": health_gauge,
        "demo_mode": True,
    }


@st.cache_data(show_spinner=False)
def get_dataset() -> object:
    try:
        return load_dataset(DATA_PATH)
    except Exception as exc:
        st.warning(f"Dataset loading hit an issue: {exc}. Using a demo dataset instead.")
        return prepare_training_dataframe(pd.DataFrame({
            "Air temperature [K]": [300.0, 301.0],
            "Process temperature [K]": [310.0, 311.0],
            "Rotational speed [rpm]": [1500.0, 1600.0],
            "Torque [Nm]": [40.0, 45.0],
            "Tool wear [min]": [100.0, 110.0],
            "Type": ["M", "L"],
            "Machine failure": [0, 1],
        }))


@st.cache_data(show_spinner=False)
def get_processed_dataset() -> object:
    return prepare_training_dataframe(get_dataset())


@st.cache_resource(show_spinner=False)
def load_or_train_model() -> dict:
    try:
        if MODEL_PATH.exists():
            bundle = load_model_bundle(MODEL_PATH)
            bundle["demo_mode"] = False
            return bundle
        with st.spinner("Training models and preparing the dashboard..."):
            bundle = train_and_save_model(DATA_PATH, MODEL_PATH)
            bundle["demo_mode"] = False
            return bundle
    except Exception as exc:
        st.warning(f"Model initialization failed: {exc}. The app is running in demo mode.")
        return build_fallback_artifact()


st.sidebar.title("Industrial Predictive Maintenance")
st.sidebar.markdown("AI-powered maintenance intelligence")
render_panel("Operations Center", "Live failure monitoring, predictive analytics, and maintenance guidance.")
page = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Data Analysis", "Model Performance", "Predict Failure", "Machine Health Dashboard"],
)

artifact = load_or_train_model()
df = get_processed_dataset()
metrics = artifact["metrics"]

if artifact.get("demo_mode"):
    st.warning("The app is running in demo mode. Train the model once to enable full predictions.")

if page == "Project Overview":
    st.markdown("<div class='hero-panel'><h1 style='margin:0;color:#f8fafc;'>Industrial Predictive Maintenance</h1><p style='margin:0.3rem 0 0 0;color:#cbd5e1;'>Premium monitoring workspace for machine health, failure prediction, and maintenance orchestration.</p></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Records", f"{len(df):,}", icon="📊", subtitle="Operational history")
    with col2:
        render_metric_card("Total Failures", f"{int(df['Machine Failure'].sum()):,}", icon="⚠️", subtitle="Critical events")
    with col3:
        failure_rate = round(df['Machine Failure'].mean() * 100, 2)
        render_metric_card("Failure Rate", f"{failure_rate}%", icon="📉", subtitle="Risk exposure")
    with col4:
        render_metric_card("Feature Count", len(artifact["feature_names"]), icon="🧠", subtitle="Model inputs")

    st.markdown("<div class='glass-panel'><h3 style='margin:0 0 0.35rem 0;'>Model Summary</h3><div style='color:#cbd5e1;'>Selected model: <b>{}</b></div></div>".format(artifact['model_name']), unsafe_allow_html=True)
    st.caption("The dashboard uses the trained classifier to estimate equipment failure probability and recommend maintenance actions.")

elif page == "Data Analysis":
    st.title("Data Analysis")
    st.markdown("Interactive visualizations for sensor behavior, quality groups, and feature relationships.")

    col1, col2 = st.columns(2)
    with col1:
        failure_counts = df['Machine Failure'].value_counts().rename(index={1: 'Failure', 0: 'Healthy'})
        failure_plot_df = pd.DataFrame({
            'Status': failure_counts.index,
            'Count': failure_counts.values,
        })
        failure_fig = px.bar(
            failure_plot_df,
            x='Status',
            y='Count',
            color='Status',
            title='Failure Distribution',
        )
        st.plotly_chart(failure_fig, width="stretch")
    with col2:
        corr = df[["Air Temperature", "Process Temperature", "Rotational Speed", "Torque", "Tool Wear", "Machine Failure"]].corr()
        corr_fig = px.imshow(corr, text_auto=True, title='Feature Correlation Heatmap')
        st.plotly_chart(corr_fig, width="stretch")

    trend_df = df[["Air Temperature", "Process Temperature", "Rotational Speed", "Torque", "Tool Wear"]].sample(200).reset_index(drop=True)
    sensor_fig = px.line(trend_df, y=trend_df.columns, title='Sensor Trends')
    st.plotly_chart(sensor_fig, width="stretch")

    product_fig = px.bar(
        df.groupby('Product Type')['Machine Failure'].mean().reset_index().rename(columns={'Machine Failure': 'Failure Rate'}),
        x='Product Type',
        y='Failure Rate',
        color='Product Type',
        title='Product Type Analysis',
    )
    st.plotly_chart(product_fig, width="stretch")

    importance_fig = px.bar(
        artifact['shap_summary'].head(10),
        x='Feature',
        y='Mean Abs SHAP',
        color='Feature',
        title='Feature Importance',
    )
    st.plotly_chart(importance_fig, width="stretch")

elif page == "Model Performance":
    st.markdown("<div class='hero-panel'><h2 style='margin:0;color:#f8fafc;'>Model Performance</h2><p style='margin:0.3rem 0 0 0;color:#cbd5e1;'>Executive-level monitoring of predictive quality and model confidence.</p></div>", unsafe_allow_html=True)
    st.dataframe(artifact['comparison_table'], width="stretch")
    if st.button("Retrain Model"):
        st.cache_resource.clear()
        st.cache_data.clear()
        with st.spinner("Retraining the predictive model..."):
            train_and_save_model(DATA_PATH, MODEL_PATH)
        st.success("Model retrained successfully.")
        st.rerun()
    st.plotly_chart(build_confusion_matrix_plot(metrics['confusion_matrix']), width="stretch")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        render_metric_card("Accuracy", f"{metrics['accuracy']:.3f}", icon="✅")
    with col2:
        render_metric_card("Precision", f"{metrics['precision']:.3f}", icon="🎯")
    with col3:
        render_metric_card("Recall", f"{metrics['recall']:.3f}", icon="🔁")
    with col4:
        render_metric_card("F1 Score", f"{metrics['f1_score']:.3f}", icon="⚖️")
    with col5:
        render_metric_card("ROC AUC", f"{metrics['roc_auc']:.3f}", icon="📈")

    st.subheader("SHAP Feature Importance")
    st.dataframe(artifact['shap_summary'].head(10), width="stretch")

elif page == "Predict Failure":
    st.markdown("<div class='hero-panel'><h2 style='margin:0;color:#f8fafc;'>Predict Failure</h2><p style='margin:0.3rem 0 0 0;color:#cbd5e1;'>Simulate operational conditions and assess maintenance urgency in real time.</p></div>", unsafe_allow_html=True)
    with st.form("prediction_form"):
        air_temp = st.number_input("Air Temperature", min_value=250.0, max_value=400.0, value=300.0)
        process_temp = st.number_input("Process Temperature", min_value=250.0, max_value=400.0, value=310.0)
        rotational_speed = st.number_input("Rotational Speed", min_value=1000.0, max_value=3000.0, value=1500.0)
        torque = st.number_input("Torque", min_value=0.0, max_value=100.0, value=40.0)
        tool_wear = st.number_input("Tool Wear", min_value=0.0, max_value=300.0, value=100.0)
        product_type = st.selectbox("Product Type", ["L", "M", "H"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "Air Temperature": air_temp,
            "Process Temperature": process_temp,
            "Rotational Speed": rotational_speed,
            "Torque": torque,
            "Tool Wear": tool_wear,
            "Product Type": product_type,
        }
        prediction = predict_failure(payload, MODEL_PATH)
        probability = prediction["failure_probability"]
        recommendation = get_maintenance_recommendation(probability)
        status = "Maintenance Required" if probability > 0.5 else "Machine Healthy"
        color = "#ef4444" if probability > 0.5 else "#22c55e"
        st.markdown(
            f"<div style='background:{color};padding:1.2rem;border-radius:1rem;color:white;font-size:1.3rem;font-weight:600'>"
            f"{status}</div>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([1.1, 0.9])
        with col1:
            st.plotly_chart(build_probability_gauge(probability), width="stretch")
        with col2:
            render_metric_card("Failure Probability", f"{probability * 100:.1f}%", icon="🔧", subtitle="Estimated risk")
            label_class = "danger" if probability > 0.8 else "warning" if probability > 0.5 else ""
            st.markdown(f"<div class='status-pill {label_class}'>{status}</div>", unsafe_allow_html=True)
            st.info(recommendation)
            report_pdf = build_prediction_report_pdf(payload, probability, status, recommendation)
            st.download_button(
                "Download Prediction Report",
                data=report_pdf,
                file_name="prediction_report.pdf",
                mime="application/pdf",
            )

elif page == "Machine Health Dashboard":
    st.markdown("<div class='hero-panel'><h2 style='margin:0;color:#f8fafc;'>Machine Health Dashboard</h2><p style='margin:0.3rem 0 0 0;color:#cbd5e1;'>Executive health view with machine score, condition status, and recommended action.</p></div>", unsafe_allow_html=True)
    sample = {
        "Air Temperature": 300.0,
        "Process Temperature": 310.0,
        "Rotational Speed": 1500.0,
        "Torque": 40.0,
        "Tool Wear": 100.0,
        "Product Type": "M",
    }
    prediction = predict_failure(sample, MODEL_PATH)
    probability = prediction["failure_probability"]
    health_score = calculate_health_score(probability)

    if health_score <= 30:
        status = "Critical"
    elif health_score <= 60:
        status = "Warning"
    else:
        status = "Healthy"

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        st.plotly_chart(artifact['health_gauge'], width="stretch")
    with col2:
        render_metric_card("Health Score", f"{health_score:.0f}/100", icon="🛠️", subtitle="0-100 machine wellness index")
        status_class = "danger" if status == "Critical" else "warning" if status == "Warning" else ""
        st.markdown(f"<div class='status-pill {status_class}'>{status}</div>", unsafe_allow_html=True)
        st.info(get_maintenance_recommendation(probability))

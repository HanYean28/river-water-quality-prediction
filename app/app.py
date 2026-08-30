"""Streamlit app for river water quality safety prediction."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DataTraining"
MODEL_DIR = DATA_DIR / "models"
TRAIN_PATH = DATA_DIR / "water_quality_train_1150.csv"
TEST_PATH = DATA_DIR / "water_quality_test_650.csv"
COMPARISON_PATH = DATA_DIR / "model_comparison_results.csv"
FEATURE_RANKING_PATH = DATA_DIR / "feature_mutual_information.csv"
ADMIN_PASSWORD = "admin123"

TARGET_COL = "is_safe"

MODEL_OPTIONS = {
    "Random Forest": "random_forest_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "SVM": "svm_model.pkl",
}

FEATURE_LABELS = {
    "aluminium": "Aluminium",
    "ammonia": "Ammonia",
    "arsenic": "Arsenic",
    "barium": "Barium",
    "cadmium": "Cadmium",
    "chloramine": "Chloramine",
    "chromium": "Chromium",
    "copper": "Copper",
    "flouride": "Flouride",
    "bacteria": "Bacteria",
    "viruses": "Viruses",
    "lead": "Lead",
    "nitrates": "Nitrates",
    "nitrites": "Nitrites",
    "mercury": "Mercury",
    "perchlorate": "Perchlorate",
    "radium": "Radium",
    "selenium": "Selenium",
    "silver": "Silver",
    "uranium": "Uranium",
}


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        input[type="password"]::-ms-reveal,
        input[type="password"]::-ms-clear {
            display: none;
        }
        .access-panel {
            text-align: center;
            padding: 0.4rem 0 1rem;
        }
        .access-icon {
            width: 52px;
            height: 52px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #eef2ff;
            color: #2563eb;
            font-size: 1.7rem;
            margin-bottom: 0.75rem;
        }
        .access-title {
            font-size: 1.45rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .access-caption {
            color: #6b7280;
            font-size: 0.92rem;
            margin-bottom: 1.2rem;
        }
        .access-choice {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            margin: 0.75rem 0 0.45rem;
            background: #ffffff;
            text-align: left;
        }
        .access-choice strong {
            display: block;
            font-size: 1.02rem;
            margin-bottom: 0.15rem;
        }
        .access-choice span {
            color: #6b7280;
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


@st.cache_data
def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_feature_ranking() -> pd.DataFrame:
    if not FEATURE_RANKING_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_RANKING_PATH)


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(MODEL_DIR / model_file)

def preload_prediction_assets(model_name: str, include_results: bool = False):
    model = load_model(MODEL_OPTIONS[model_name])
    load_train_data()
    load_test_data()
    get_feature_medians()
    get_feature_bounds()
    if include_results:
        load_model_comparison()
        load_feature_ranking()
    return model


@st.cache_data
def get_feature_medians() -> dict[str, float]:
    train_df = load_train_data()
    return train_df.drop(columns=[TARGET_COL]).median(numeric_only=True).to_dict()


@st.cache_data
def get_feature_bounds() -> dict[str, tuple[float, float]]:
    train_df = load_train_data()
    features = train_df.drop(columns=[TARGET_COL])
    return {
        column: (float(features[column].min()), float(features[column].max()))
        for column in features.columns
    }


def get_feature_columns() -> list[str]:
    train_df = load_train_data()
    return [column for column in train_df.columns if column != TARGET_COL]


def build_feature_row(user_inputs: dict[str, float]) -> pd.DataFrame:
    feature_columns = get_feature_columns()
    medians = get_feature_medians()
    row = {feature: float(user_inputs.get(feature, medians[feature])) for feature in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def predict(model, features: pd.DataFrame) -> int:
    return int(model.predict(features)[0])


def format_label(value: int) -> str:
    return "Safe" if int(value) == 1 else "Unsafe"


def initialize_state() -> None:
    defaults = {
        "access_mode": None,
        "admin_authenticated": False,
        "login_error": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_access() -> None:
    st.session_state.access_mode = None
    st.session_state.admin_authenticated = False
    st.session_state.login_error = False


def choose_guest_access() -> None:
    st.session_state.access_mode = "guest"


def choose_admin_access() -> None:
    st.session_state.access_mode = "admin"
    st.session_state.admin_authenticated = False
    st.session_state.login_error = False


def authenticate_admin() -> None:
    if st.session_state.get("admin_access_code") == ADMIN_PASSWORD:
        st.session_state.admin_authenticated = True
        st.session_state.login_error = False
    else:
        st.session_state.login_error = True



def render_access_selection() -> None:
    st.title("River Water Quality Prediction")
    st.write("Predict whether a water sample is safe or unsafe using Random Forest, Decision Tree, or SVM.")

    left, center, right = st.columns([1, 0.75, 1])
    with center:
        st.subheader("Choose Access")

        with st.container(border=True):
            st.markdown("**Guest Access**")
            st.caption("Use manual input and test samples for prediction only.")
            st.button("Continue as Guest", width="stretch", on_click=choose_guest_access)

        with st.container(border=True):
            st.markdown("**Admin Login**")
            st.caption("Access prediction tools and model performance results.")
            st.button("Continue as Admin", width="stretch", on_click=choose_admin_access)

def render_model_sidebar(show_target_label: bool = True, allow_model_selection: bool = True):
    with st.sidebar:
        st.header("Model")
        if allow_model_selection:
            model_name = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()), index=0)
        else:
            model_name = "Random Forest"
            st.write("Random Forest")
            st.caption("Highest accuracy model")

        if show_target_label:
            st.header("Target Label")
            st.write("0 = Unsafe")
            st.write("1 = Safe")

        st.divider()
        st.button("Back to access selection", on_click=reset_access)

    return model_name


def render_model_metrics(model_name: str) -> None:
    comparison = load_model_comparison()
    if comparison.empty:
        return

    row = comparison[comparison["Model"] == model_name]
    if row.empty:
        return

    row = row.iloc[0]
    cols = st.columns(4)
    cols[0].metric(
        "Accuracy",
        f"{row['Accuracy'] * 100:.2f}%",
        help="Overall percentage of correct predictions on the test set.",
    )
    cols[1].metric(
        "Precision",
        f"{row['Precision_Safe_Class_1'] * 100:.2f}%",
        help="Of the samples predicted as safe, how many were actually safe.",
    )
    cols[2].metric(
        "Recall",
        f"{row['Recall_Safe_Class_1'] * 100:.2f}%",
        help="Of the actual safe samples, how many were correctly predicted as safe.",
    )
    cols[3].metric(
        "F1",
        f"{row['F1_Safe_Class_1'] * 100:.2f}%",
        help="Balance between precision and recall.",
    )


def render_manual_inputs() -> dict[str, float]:
    feature_columns = get_feature_columns()
    medians = get_feature_medians()
    bounds = get_feature_bounds()

    st.info(
        "Input validation is applied using the observed training-data range. "
        "Values outside this range are blocked because the model has not learned from such values."
    )

    inputs: dict[str, float] = {}
    cols = st.columns(3)
    for index, feature in enumerate(feature_columns):
        min_value, max_value = bounds[feature]
        step = 0.001 if max_value <= 1 else 0.01
        with cols[index % 3]:
            inputs[feature] = st.number_input(
                feature_label(feature),
                min_value=min_value,
                max_value=max_value,
                value=float(medians[feature]),
                step=step,
                format="%.4f",
                help=f"Allowed range based on training data: {min_value:.4f} to {max_value:.4f}",
            )
    return inputs


def show_prediction(prediction: int, actual_label: int | None = None) -> None:
    if prediction == 1:
        st.success("Prediction: Safe water")
    else:
        st.error("Prediction: Unsafe water")

    if actual_label is not None:
        actual_text = format_label(actual_label)
        result = "Correct" if prediction == int(actual_label) else "Incorrect"
        st.write(f"Actual label: **{actual_text}**")
        st.write(f"Model result: **{result}**")


def render_sample_table(record: pd.Series, show_actual: bool) -> None:
    feature_columns = get_feature_columns()
    rows = {
        "Feature": [feature_label(feature) for feature in feature_columns],
        "Value": [f"{record[feature]:.4f}" for feature in feature_columns],
    }
    if show_actual:
        rows["Feature"].append("Actual Result")
        rows["Value"].append(format_label(record[TARGET_COL]))

    table = pd.DataFrame(rows).astype(str)
    st.dataframe(table, hide_index=True, width="stretch")


def render_performance_chart(comparison: pd.DataFrame) -> None:
    metric_cols = {
        "Accuracy": "Accuracy",
        "Precision_Safe_Class_1": "Precision",
        "Recall_Safe_Class_1": "Recall",
        "F1_Safe_Class_1": "F1 Score",
    }
    graph_df = comparison[["Model", *metric_cols.keys()]].rename(columns=metric_cols)
    graph_df = graph_df.melt(id_vars="Model", var_name="Metric", value_name="Metric Value")
    graph_df["Metric Value"] = graph_df["Metric Value"] * 100

    chart = (
        alt.Chart(graph_df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", title="Classification Model"),
            y=alt.Y("Metric Value:Q", title="Metric Value (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Metric:N", title="Metric"),
            xOffset="Metric:N",
            tooltip=["Model", "Metric", alt.Tooltip("Metric Value:Q", format=".2f")],
        )
        .properties(title="Classification Model Performance Comparison", height=420)
    )
    st.altair_chart(chart, width="stretch")




def build_confusion_matrix_data(model_name: str) -> pd.DataFrame:
    test_df = load_test_data()
    model = load_model(MODEL_OPTIONS[model_name])
    feature_columns = get_feature_columns()

    y_true = test_df[TARGET_COL].astype(int)
    y_pred = pd.Series(model.predict(test_df[feature_columns]).astype(int), index=test_df.index)

    return pd.DataFrame(
        [
            {
                "Model": model_name,
                "Actual": "Unsafe",
                "Predicted": "Unsafe",
                "Count": int(((y_true == 0) & (y_pred == 0)).sum()),
            },
            {
                "Model": model_name,
                "Actual": "Unsafe",
                "Predicted": "Safe",
                "Count": int(((y_true == 0) & (y_pred == 1)).sum()),
            },
            {
                "Model": model_name,
                "Actual": "Safe",
                "Predicted": "Unsafe",
                "Count": int(((y_true == 1) & (y_pred == 0)).sum()),
            },
            {
                "Model": model_name,
                "Actual": "Safe",
                "Predicted": "Safe",
                "Count": int(((y_true == 1) & (y_pred == 1)).sum()),
            },
        ]
    )


def render_confusion_matrix(model_name: str) -> None:
    matrix_df = build_confusion_matrix_data(model_name)
    max_count = matrix_df["Count"].max()

    heatmap = (
        alt.Chart(matrix_df)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X("Predicted:N", title="Predicted Label", sort=["Unsafe", "Safe"]),
            y=alt.Y("Actual:N", title="Actual Label", sort=["Unsafe", "Safe"]),
            color=alt.Color(
                "Count:Q",
                title="Samples",
                scale=alt.Scale(scheme="blues", domain=[0, max_count]),
            ),
            tooltip=["Actual", "Predicted", "Count"],
        )
    )
    labels = (
        alt.Chart(matrix_df)
        .mark_text(fontSize=18, fontWeight="bold")
        .encode(
            x=alt.X("Predicted:N", sort=["Unsafe", "Safe"]),
            y=alt.Y("Actual:N", sort=["Unsafe", "Safe"]),
            text="Count:Q",
            color=alt.condition(
                alt.datum.Count > max_count * 0.55,
                alt.value("white"),
                alt.value("#111827"),
            ),
        )
    )

    left, center, right = st.columns([1, 1.15, 1])
    with center:
        st.markdown(f"**{model_name}**")
        st.altair_chart((heatmap + labels).properties(width=460, height=320), width="content")


def render_results(model_name: str) -> None:
    comparison = load_model_comparison()
    ranking = load_feature_ranking()

    st.subheader(f"{model_name} Performance")
    render_model_metrics(model_name)

    if not comparison.empty:
        st.subheader("Model Comparison")
        render_performance_chart(comparison)

        display_df = comparison.copy()
        percent_cols = [
            "Accuracy",
            "Precision_Safe_Class_1",
            "Recall_Safe_Class_1",
            "F1_Safe_Class_1",
            "Recall_Unsafe_Class_0",
        ]
        display_df[percent_cols] = (display_df[percent_cols] * 100).round(2)
        display_df = display_df.rename(columns={
            "Accuracy": "Accuracy (%)",
            "Precision_Safe_Class_1": "Precision Safe (%)",
            "Recall_Safe_Class_1": "Recall Safe (%)",
            "F1_Safe_Class_1": "F1 Safe (%)",
            "Recall_Unsafe_Class_0": "Recall Unsafe (%)",
            "False_Safe_Count": "False Safe",
            "False_Unsafe_Count": "False Unsafe",
        })
        left, center, right = st.columns([0.05, 1.9, 0.05])
        with center:
            st.dataframe(display_df, hide_index=True, width="stretch")

        st.subheader("Confusion Matrix")
        render_confusion_matrix(model_name)

    if not ranking.empty:
        st.subheader("Mutual Information Feature Ranking")
        ranking_display = ranking.copy()
        ranking_display["Mutual Information"] = ranking_display["Mutual Information"].round(4)

        chart_df = ranking_display.sort_values("Mutual Information", ascending=False)
        chart_height = max(420, len(chart_df) * 28)
        base_chart = alt.Chart(chart_df).encode(
            x=alt.X("Mutual Information:Q", title="Mutual Information"),
            y=alt.Y("Feature:N", sort="-x", title="Feature"),
            tooltip=["Rank", "Feature", alt.Tooltip("Mutual Information:Q", format=".4f")],
        )
        bars = base_chart.mark_bar(color="#2563eb")
        labels = base_chart.mark_text(
            align="left",
            baseline="middle",
            dx=4,
            color="#111827",
        ).encode(text=alt.Text("Mutual Information:Q", format=".4f"))
        chart = (bars + labels).properties(height=chart_height)
        st.altair_chart(chart, width="stretch")


def render_prediction_tabs(model, show_actual: bool, include_results: bool, model_name: str) -> None:
    if include_results:
        tab_manual, tab_sample, tab_results = st.tabs(["Manual Input", "Test Sample", "Results"])
    else:
        tab_manual, tab_sample = st.tabs(["Manual Input", "Test Sample"])
        tab_results = None

    with tab_manual:
        st.subheader("Manual Water Quality Input")
        user_inputs = render_manual_inputs()
        if st.button("Predict Water Safety", type="primary"):
            features = build_feature_row(user_inputs)
            prediction = predict(model, features)
            show_prediction(prediction)

    with tab_sample:
        st.subheader("Test Set Sample")
        test_df = load_test_data()
        sample_index = st.number_input(
            "Sample index",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            step=1,
        )
        record = test_df.iloc[int(sample_index)]
        render_sample_table(record, show_actual=show_actual)

        if st.button("Predict Selected Sample", type="primary"):
            features = pd.DataFrame([record[get_feature_columns()].to_dict()], columns=get_feature_columns())
            prediction = predict(model, features)
            actual = int(record[TARGET_COL]) if show_actual else None
            show_prediction(prediction, actual_label=actual)

    if tab_results is not None:
        with tab_results:
            render_results(model_name)


def render_guest_view() -> None:
    model_name = "Random Forest"
    with st.spinner("Loading prediction system..."):
        model = preload_prediction_assets(model_name)

    render_model_sidebar(show_target_label=False, allow_model_selection=False)
    st.title("River Water Quality Prediction")
    st.write("Guest access: use manual input or test samples to get a prediction.")
    render_prediction_tabs(model, show_actual=False, include_results=False, model_name=model_name)


def render_admin_view() -> None:

    if not st.session_state.admin_authenticated:
        left, center, right = st.columns([1, 1.1, 1])
        with center:
            st.subheader("Admin Login")
            st.text_input(
                "Admin access code",
                key="admin_access_code",
                type="password",
                autocomplete="off",
                placeholder="Enter admin access code",
            )
            st.button("Login", type="primary", width="stretch", on_click=authenticate_admin)
            if st.session_state.login_error:
                st.error("Incorrect access code")
        return

    model_name = render_model_sidebar(show_target_label=True, allow_model_selection=True)
    with st.spinner("Loading admin dashboard..."):
        model = preload_prediction_assets(model_name, include_results=True)

    st.title("River Water Quality Prediction")
    st.write("Admin access: prediction tools and model performance results.")
    render_prediction_tabs(model, show_actual=True, include_results=True, model_name=model_name)


def main() -> None:
    st.set_page_config(page_title="River Water Quality Prediction", layout="wide")
    apply_custom_css()
    initialize_state()

    missing_files = [
        path for path in [TRAIN_PATH, TEST_PATH, *[MODEL_DIR / file for file in MODEL_OPTIONS.values()]]
        if not path.exists()
    ]
    if missing_files:
        st.error("Some required files are missing. Run the preprocessing and model training notebooks first.")
        for path in missing_files:
            st.write(path)
        return

    if st.session_state.access_mode is None:
        render_access_selection()
    elif st.session_state.access_mode == "guest":
        render_guest_view()
    elif st.session_state.access_mode == "admin":
        render_admin_view()


if __name__ == "__main__":
    main()

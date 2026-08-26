"""Streamlit app for river water quality safety prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DataTraining"
MODEL_DIR = DATA_DIR / "models"
TRAIN_PATH = DATA_DIR / "water_quality_train_1150.csv"
TEST_PATH = DATA_DIR / "water_quality_test_650.csv"
COMPARISON_PATH = DATA_DIR / "model_comparison_results.csv"
FEATURE_RANKING_PATH = DATA_DIR / "feature_mutual_information.csv"

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


def get_prediction_confidence(model, features: pd.DataFrame, prediction: int) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)[0]
    class_index = list(model.classes_).index(prediction)
    return float(probabilities[class_index])


def format_label(value: int) -> str:
    return "Safe" if int(value) == 1 else "Unsafe"


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


def show_prediction(
    prediction: int,
    confidence: float | None = None,
    actual_label: int | None = None,
) -> None:
    if prediction == 1:
        st.success("Prediction: Safe water")
    else:
        st.error("Prediction: Unsafe water")

    if confidence is not None:
        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%",
            help="Probability assigned by the model to the predicted class. This is not a guarantee that the prediction is correct.",
        )

    if actual_label is not None:
        actual_text = format_label(actual_label)
        result = "Correct" if prediction == int(actual_label) else "Incorrect"
        st.write(f"Actual label: **{actual_text}**")
        st.write(f"Model result: **{result}**")


def render_sample_table(record: pd.Series) -> None:
    feature_columns = get_feature_columns()
    table = pd.DataFrame({
        "Feature": [feature_label(feature) for feature in feature_columns] + ["Actual Result"],
        "Value": [f"{record[feature]:.4f}" for feature in feature_columns] + [format_label(record[TARGET_COL])],
    })
    st.dataframe(table, hide_index=True, width="stretch")


def render_results(model_name: str) -> None:
    comparison = load_model_comparison()
    ranking = load_feature_ranking()

    st.subheader(f"{model_name} Performance")
    render_model_metrics(model_name)

    if not comparison.empty:
        st.subheader("Model Comparison")
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
            "Precision_Safe_Class_1": "Precision Safe (%)",
            "Recall_Safe_Class_1": "Recall Safe (%)",
            "F1_Safe_Class_1": "F1 Safe (%)",
            "Recall_Unsafe_Class_0": "Recall Unsafe (%)",
            "False_Safe_Count": "False Safe",
            "False_Unsafe_Count": "False Unsafe",
            "Accuracy": "Accuracy (%)",
        })
        st.dataframe(display_df, hide_index=True, width="stretch", height=150)

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



def main() -> None:
    st.set_page_config(page_title="River Water Quality Prediction", layout="wide")

    st.title("River Water Quality Prediction")
    st.write("Predict whether a water sample is safe or unsafe using Random Forest, Decision Tree, or SVM.")

    missing_files = [
        path for path in [TRAIN_PATH, TEST_PATH, *[MODEL_DIR / file for file in MODEL_OPTIONS.values()]]
        if not path.exists()
    ]
    if missing_files:
        st.error("Some required files are missing. Run the preprocessing and model training notebooks first.")
        for path in missing_files:
            st.write(path)
        return

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()), index=1)
        model = load_model(MODEL_OPTIONS[model_name])

        st.header("Target Label")
        st.write("0 = Unsafe")
        st.write("1 = Safe")

    tab_manual, tab_sample, tab_results = st.tabs(["Manual Input", "Test Sample", "Results"])

    with tab_manual:
        st.subheader("Manual Water Quality Input")
        user_inputs = render_manual_inputs()
        if st.button("Predict Water Safety", type="primary"):
            features = build_feature_row(user_inputs)
            prediction = predict(model, features)
            confidence = get_prediction_confidence(model, features, prediction)
            show_prediction(prediction, confidence=confidence)

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
        render_sample_table(record)

        if st.button("Predict Selected Sample", type="primary"):
            features = pd.DataFrame([record[get_feature_columns()].to_dict()], columns=get_feature_columns())
            prediction = predict(model, features)
            confidence = get_prediction_confidence(model, features, prediction)
            show_prediction(prediction, confidence=confidence, actual_label=int(record[TARGET_COL]))

    with tab_results:
        render_results(model_name)


if __name__ == "__main__":
    main()












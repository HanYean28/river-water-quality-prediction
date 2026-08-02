"""Streamlit demo for sepsis prediction within a 6-hour window."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "DataTraining" / "models"
DATASET_PATH = ROOT / "Sepsis_dataset.csv"

TARGET_COL = "SepsisLabel"

FEATURE_COLUMNS = [
    "Hour", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS",
]

KEY_VITALS = [
    "Hour", "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "Lactate", "WBC", "Age", "Gender", "ICULOS",
]

# Training-set medians (same as AI_Assignment.ipynb Section 2.3)
FEATURE_MEDIANS = {
    "Hour": 20.0,
    "HR": 84.0,
    "O2Sat": 98.0,
    "Temp": 37.06,
    "SBP": 119.0,
    "MAP": 77.0,
    "DBP": 59.0,
    "Resp": 18.0,
    "BaseExcess": 0.0,
    "HCO3": 24.0,
    "FiO2": 0.5,
    "pH": 7.39,
    "PaCO2": 40.0,
    "SaO2": 97.0,
    "AST": 57.0,
    "BUN": 18.0,
    "Alkalinephos": 79.0,
    "Calcium": 8.3,
    "Chloride": 106.0,
    "Creatinine": 0.9,
    "Bilirubin_direct": 1.5,
    "Glucose": 124.0,
    "Lactate": 1.8,
    "Magnesium": 2.0,
    "Phosphate": 3.4,
    "Potassium": 4.1,
    "Bilirubin_total": 0.9,
    "TroponinI": 4.75,
    "Hct": 30.2,
    "Hgb": 10.4,
    "PTT": 32.4,
    "WBC": 10.8,
    "Fibrinogen": 248.0,
    "Platelets": 181.0,
    "Age": 65.27,
    "Gender": 1.0,
    "Unit1": 1.0,
    "Unit2": 0.0,
    "HospAdmTime": -2.59,
    "ICULOS": 21.0,
}

MODEL_OPTIONS = {
    "KNN (recommended)": "knn_model.pkl",
    "SVM": "svm_model.pkl",
    "Decision Tree": "dt_model.pkl",
}

VITAL_BOUNDS = {
    "Hour": (0, 48, 1),
    "HR": (40, 180, 1),
    "O2Sat": (70, 100, 1),
    "Temp": (35.0, 41.0, 0.1),
    "SBP": (60, 200, 1),
    "MAP": (40, 130, 1),
    "DBP": (30, 120, 1),
    "Resp": (8, 40, 1),
    "Lactate": (0.5, 15.0, 0.1),
    "WBC": (1.0, 40.0, 0.1),
    "Age": (18.0, 95.0, 0.1),
    "Gender": (0, 1, 1),
    "ICULOS": (1, 100, 1),
}


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning steps as AI_Assignment.ipynb Section 2."""
    cleaned = df.drop_duplicates()
    cleaned = cleaned.drop(columns=["Patient_ID", "Unnamed: 0"], errors="ignore")
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.dropna(subset=[TARGET_COL])

    medians = cleaned.drop(columns=[TARGET_COL]).median(numeric_only=True)
    cleaned = cleaned.copy()
    cleaned[medians.index] = cleaned[medians.index].fillna(medians)
    return cleaned


def build_feature_row(user_inputs: dict | None = None) -> pd.DataFrame:
    """Build one model-ready row; missing fields use training medians."""
    row = pd.Series(FEATURE_MEDIANS, dtype=float)
    if user_inputs:
        for key, value in user_inputs.items():
            if key in row.index:
                row[key] = value
    return pd.DataFrame([row[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)


def row_from_cleaned_record(record: pd.Series) -> pd.DataFrame:
    values = record[FEATURE_COLUMNS].astype(float).values
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(MODEL_DIR / model_file)


@st.cache_data
def get_sample_patients() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    cleaned = clean_raw_dataframe(df)
    return cleaned.sample(n=min(200, len(cleaned)), random_state=42)


def predict(model, features: pd.DataFrame) -> tuple[int, float | None]:
    prediction = int(model.predict(features)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(features)[0][1])
    return prediction, probability


def render_vital_inputs() -> dict:
    inputs: dict = {}
    cols = st.columns(3)
    for idx, vital in enumerate(KEY_VITALS):
        min_val, max_val, step = VITAL_BOUNDS[vital]
        default = float(FEATURE_MEDIANS[vital])
        with cols[idx % 3]:
            if vital == "Gender":
                inputs[vital] = st.selectbox(
                    "Gender",
                    options=[0, 1],
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    index=int(default),
                )
            else:
                inputs[vital] = st.number_input(
                    vital,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=default,
                    step=float(step),
                )
    return inputs


def show_result(prediction: int, probability: float | None, actual_label: int | None = None) -> None:
    if prediction == 1:
        st.error("Prediction: **Sepsis likely within 6 hours**")
    else:
        st.success("Prediction: **No sepsis predicted within 6 hours**")

    if probability is not None:
        st.metric("Sepsis probability", f"{probability * 100:.1f}%")
    else:
        st.caption("This model does not output probability scores (SVM).")

    if actual_label is not None:
        actual_text = "Sepsis" if actual_label == 1 else "No sepsis"
        match = "Correct" if actual_label == prediction else "Incorrect"
        st.write(f"Actual label: **{actual_text}** — model was **{match}** on this sample.")


def main() -> None:
    st.set_page_config(page_title="Sepsis Prediction", page_icon="🩺", layout="wide")

    st.title("Sepsis Prediction Demo")
    st.caption(
        "BMCS Artificial Intelligence — Predicting sepsis development within a 6-hour window. "
        "For educational demo only; not for clinical use."
    )

    with st.sidebar:
        st.header("Model")
        model_label = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
        model = load_model(MODEL_OPTIONS[model_label])
        st.info("KNN has the highest recall (66.9%) on the test set.")

        st.header("About")
        st.markdown(
            "- Missing lab values are filled with **training medians** (same as notebook).\n"
            "- Each saved model includes **scaler + classifier**.\n"
            "- Test set remains imbalanced; the UI shows risk, not a diagnosis."
        )

    tab_manual, tab_sample = st.tabs(["Enter vitals", "Try sample patient"])

    with tab_manual:
        st.subheader("Key vital signs")
        st.write("Adjust the main vitals below. Other features use training-set medians automatically.")
        user_inputs = render_vital_inputs()

        with st.expander("Advanced: edit all 40 features"):
            advanced_inputs = {}
            adv_cols = st.columns(4)
            for idx, col in enumerate(FEATURE_COLUMNS):
                with adv_cols[idx % 4]:
                    advanced_inputs[col] = st.number_input(
                        col,
                        value=float(user_inputs.get(col, FEATURE_MEDIANS[col])),
                        format="%.4f",
                        key=f"adv_{col}",
                    )
            user_inputs = advanced_inputs

        if st.button("Predict sepsis risk", type="primary", key="predict_manual"):
            features = build_feature_row(user_inputs)
            prediction, probability = predict(model, features)
            show_result(prediction, probability)

    with tab_sample:
        st.subheader("Random patient from dataset")
        samples = get_sample_patients()
        sample_idx = st.number_input("Sample index", min_value=0, max_value=len(samples) - 1, value=0, step=1)
        record = samples.iloc[int(sample_idx)]
        actual = int(record[TARGET_COL])

        st.dataframe(
            record[FEATURE_COLUMNS + [TARGET_COL]].astype(float).to_frame("Value")
        )

        if st.button("Predict for this patient", type="primary", key="predict_sample"):
            features = row_from_cleaned_record(record)
            prediction, probability = predict(model, features)
            show_result(prediction, probability, actual_label=actual)


if __name__ == "__main__":
    main()

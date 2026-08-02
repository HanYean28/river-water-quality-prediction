# Sepsis Prediction

Predict sepsis development within a 6-hour window using SVM, KNN, and Decision Tree classifiers.

## Project structure

```
sepsis-prediction/
├── AI_Assignment.ipynb      # Main notebook (EDA, training, evaluation)
├── Sepsis_dataset.csv       # Dataset
├── app/
│   └── app.py               # Streamlit UI + preprocessing (all-in-one)
├── DataTraining/
│   └── models/
│       ├── svm_model.pkl
│       ├── knn_model.pkl
│       └── dt_model.pkl
└── requirements.txt
```

## Run the Streamlit app

From the project root:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app/app.py
```

The app opens in your browser. Use **KNN (recommended)** — it had the best recall on the test set.

## Notes

- Each `.pkl` file contains a `GridSearchCV` object with **scaler + model** inside a Pipeline.
- Missing inputs in the UI are filled with training-set medians (defined in `app/app.py`).
- This demo is for educational purposes only, not clinical diagnosis.

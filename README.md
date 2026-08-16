# River Water Quality Prediction

Predict whether river water is safe or unsafe using supervised machine learning classifiers.

## Dataset

The project uses `waterQuality.csv`, which contains water quality measurements such as aluminium, ammonia, arsenic, bacteria, viruses, lead, nitrates, mercury, and other chemical indicators.

Target column:

- `is_safe`
  - `1` = safe water
  - `0` = unsafe water

## Project Structure

```text
river-water-quality-prediction/
|-- Preprocessing.ipynb      # Data cleaning, balancing, splitting, feature selection
|-- ModelTraining.ipynb      # KNN, Decision Tree, SVM training and evaluation
|-- waterQuality.csv         # Original water quality dataset
|-- app/
|   `-- app.py               # Streamlit prediction app
|-- DataTraining/            # Preprocessed data, results, and trained models
`-- requirements.txt
```

## Run The Streamlit App

From the project root:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app/app.py
```

## Notes

- This is a supervised binary classification problem.
- The project uses `is_safe` as the prediction target.
- The final app uses the selected features saved in the train/test datasets.

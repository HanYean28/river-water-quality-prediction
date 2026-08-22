# River Water Quality Prediction

A supervised machine learning project that predicts whether river water is safe or unsafe based on water quality measurements. The project includes data preprocessing, feature selection, model training, model evaluation, and a Streamlit web application for manual prediction.

## Project Overview

This project solves a binary classification problem using the target column `is_safe`:

- `1` = Safe water
- `0` = Unsafe water

Three supervised classification models are trained and compared:

- Random Forest
- Decision Tree
- Support Vector Machine (SVM)

The final application allows users to enter water quality values manually or test existing samples from the test dataset.

## Dataset

The project uses `waterQuality.csv`, which contains water quality measurements such as:

- aluminium
- cadmium
- chromium
- arsenic
- chloramine
- perchlorate
- radium
- nitrates
- bacteria
- viruses
- uranium

The original dataset is cleaned and prepared before model training. In the final cleaned dataset, no rows were removed for missing feature values because no remaining feature values were missing.

## Preprocessing Steps

The preprocessing workflow is implemented in `Preprocessing.ipynb`.

1. Load the original dataset from `waterQuality.csv`.
2. Convert all columns to numeric format.
3. Remove rows with invalid or missing target values.
4. Remove duplicate rows.
5. Remove rows with missing feature values if any are found.
6. Balance the dataset to contain equal target classes:
   - 900 safe samples
   - 900 unsafe samples
   - 1800 total samples
7. Split the balanced dataset into:
   - 1150 training records
   - 650 testing records
8. Apply Mutual Information feature selection using the training data only.
9. Remove features with Mutual Information score equal to 0.
10. Save the final preprocessed train/test datasets.

Feature selection is performed after the train/test split to avoid data leakage. The selected features are then applied to both training and testing datasets.

## Feature Selection

This project uses Mutual Information as a filter-based feature selection method.

Selected features:

- aluminium
- cadmium
- chromium
- arsenic
- chloramine
- perchlorate
- radium
- nitrates
- barium
- bacteria
- silver
- selenium
- ammonia
- viruses
- uranium

Removed features:

- copper
- lead
- flouride
- mercury
- nitrites

## Model Training

Model training and evaluation are implemented in `ModelTraining.ipynb`.

The models are trained using the preprocessed training dataset and evaluated using the separate test dataset. Hyperparameter tuning is performed using `GridSearchCV` with 5-fold cross-validation.

## Model Results

Final test results using the 650-record test set:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Random Forest | 93.23% | 93.77% | 92.62% | 93.19% |
| Decision Tree | 90.15% | 91.69% | 88.31% | 89.97% |
| SVM | 89.38% | 89.51% | 89.23% | 89.37% |

Random Forest achieved the highest accuracy and was selected as the best-performing model.

## Streamlit Application

The Streamlit app is located in `app/app.py`.

Main features:

- Select trained model
- Enter manual water quality values
- Predict whether water is safe or unsafe
- Test prediction using records from the test set
- View model performance results
- View Mutual Information feature ranking graph

## Project Structure

```text
river-water-quality-prediction/
|-- Preprocessing.ipynb                 # Data cleaning, balancing, splitting, feature selection
|-- ModelTraining.ipynb                 # Model training, hyperparameter tuning, evaluation
|-- waterQuality.csv                    # Original dataset
|-- README.md                           # Project documentation
|-- requirements.txt                    # Python dependencies
|-- app/
|   `-- app.py                          # Streamlit web application
|-- DataTraining/
|   |-- water_quality_cleaned.csv
|   |-- water_quality_balanced_1800.csv
|   |-- water_quality_train_1150.csv
|   |-- water_quality_test_650.csv
|   |-- feature_mutual_information.csv
|   |-- model_comparison_results.csv
|   `-- models/
|       |-- random_forest_model.pkl
|       |-- decision_tree_model.pkl
|       `-- svm_model.pkl
```

## How To Run Locally

Open a terminal in the project folder:

```powershell
cd river-water-quality-prediction
```

Install the required packages:

```powershell
py -3 -m pip install -r requirements.txt
```

Run the Streamlit app:

```powershell
py -3 -m streamlit run app/app.py
```

Open the local app in a browser:

```text
http://localhost:8501
```

## Notes

- This is a supervised binary classification project.
- The final dataset is balanced before model training.
- Mutual Information feature selection is calculated using training data only to avoid data leakage.
- The prediction result should be interpreted as a machine learning prediction, not as a guaranteed real-world water safety decision.


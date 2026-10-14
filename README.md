# Bazel Build-Time Prediction Using Machine Learning

## Project Overview

This mini-project reproduces the reference study *Predicting Bazel Build Times Using Machine Learning* and extends it with a Random Forest Regressor.

The objective is to predict Bazel build CPU time in milliseconds using categorical information about the most common file-path prefix and file type.

The project has two parts:

- **Part 1 — Reference Reproduction:** Linear Regression trained using mini-batch Stochastic Gradient Descent (SGD), with and without prefix × type feature crossing.
- **Part 2 — Model Extension:** Random Forest Regressor evaluated against the reference models.

A Streamlit application provides an interactive interface for estimating build CPU time.

## Dataset

The project uses the supplied files:

- `InputData.csv` — file-path prefixes and file types for 500 examples.
- `CPUTimes.csv` — CPU-time targets in milliseconds for 500 examples.

The two files are aligned by row order because the supplied files do not contain a shared commit identifier.

The experiments use the following split:

| Split | Records |
|---|---:|
| Training | 300 |
| Validation | 100 |
| Testing | 100 |
| Total | 500 |

## Models and Methodology

### Part 1: Reference Reproduction

Two linear-regression experiments are implemented:

1. Linear Regression without feature crossing.
2. Linear Regression with prefix × type feature crossing.

The implementation uses:

- Learning rate: `0.01`
- Mini-batch size: `16`
- Training periods: `9`
- Evaluation metric: Root Mean Squared Error (RMSE)

Training and validation RMSE are recorded across all nine periods. Reproduced training RMSE values are compared with the reference paper's reported values.

### Part 2: Random Forest Extension

A Random Forest Regressor is added as a model that differs from the reference's linear-regression approach.

Configuration:

- Number of trees: `300`
- Random state: `42`
- Maximum features: `1.0`
- Minimum samples per leaf: `1`

The model uses the same training, validation and test split and is evaluated using RMSE. Its test results are compared with those of both linear-regression models.

## Requirements

- Python 3
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Streamlit

## Setup Instructions

Open a terminal in the project directory.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib streamlit
```

## How to Run

### Run Part 1

```bash
python part1_reproduction.py
```

This generates the nine-period results, the reported-versus-reproduced comparison, and the training and validation RMSE plots.

### Run Part 2

Run Part 1 first, then execute:

```bash
python part2_random_forest.py
```

This generates Random Forest evaluation results, test predictions, a model-comparison table, and a feature-importance plot.

### Run the Streamlit Predictor

```bash
streamlit run app.py
```

Select a file-path prefix and file type, then click **Predict Build Time** to view the estimated CPU time.

The application trains the Random Forest using the first 300 records and uses the same categorical feature representation as the extension experiment.

## Generated Outputs

### Part 1

- `part1_period_results.csv`
- `part1_reported_vs_yours.csv`
- `part1_training_rmse.png`
- `part1_validation_rmse.png`

### Part 2

- `part2_random_forest_results.csv`
- `part2_random_forest_test_predictions.csv`
- `part2_model_comparison.csv`
- `part2_random_forest_feature_importance.png`

## Limitations

- The dataset contains only 500 examples.
- The supplied input and target files are aligned by row order because a shared commit identifier is unavailable.
- The paper does not specify every detail of the data-splitting procedure and hyperparameter search.
- Random Forest does not necessarily outperform the linear models on this dataset.
- The Streamlit application provides an experimental estimate based on the available categorical features; actual Bazel build times can differ.

## Reference

Nadeem, Muhammad Umar, and Sarah Raza. *Predicting Bazel Build Times Using Machine Learning*.

This project is an academic reproduction and extension of the reference study.
"""
Part 2 — Extension with a Random Forest Regressor

Reference project:
    "Predicting Bazel Build Times Using Machine Learning"

This file implements ONE additional regression model that is not used in the
reference paper. It deliberately does not import or reuse the authors'
Python implementation.

Fair-comparison design:
- same 500-row supplied dataset
- same row-order split used in Part 1: 300 train / 100 validation / 100 test
- same target: CPUTime (milliseconds)
- same categorical preprocessing used in Part 1
- same prefix × type feature-crossing representation used by the stronger
  reference experiment
- same evaluation metric: RMSE

Additional model:
    RandomForestRegressor

Chosen hyperparameters:
    n_estimators = 300
    random_state = 42
    n_jobs = -1
    max_features = 1.0
    min_samples_leaf = 1

Why Random Forest?
The reference uses linear regression. A random forest provides a clearly
different nonlinear model family that can represent interactions between
categorical features without requiring the linear model itself to be the
learning mechanism. It is also practical for the small (500-example),
categorical/tabular dataset used in the reference.

Important comparison note:
The reference paper's reported 5735.84 ms and 5628.09 ms figures are averages
of TRAINING RMSE values over nine periods. For a fair Part 2 model comparison,
this file compares models on the SAME 100-example TEST SPLIT instead. The
Linear Regression numbers in the comparison table are the final (period 9)
test RMSE values from Part 1.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------- Configuration -----------------------------

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "InputData.csv"
TARGET_FILE = DATA_DIR / "CPUTimes.csv"
PART1_RESULTS_FILE = DATA_DIR / "part1_period_results.csv"

N_TRAIN = 300
N_VALIDATION = 100
N_TEST = 100

RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_FEATURES = 1.0
MIN_SAMPLES_LEAF = 1

PREFIX_VOCAB = [
    "bazelci/",
    "examples/",
    "scripts/",
    "site/",
    "src/conditions",
    "src/java_tools",
    "src/main",
    "src/test",
    "src/tools",
    "third_party/",
    "tools/",
]

TYPE_VOCAB = [
    "JAVA",
    "C/C++",
    "Starlark",
    "python",
    "HTML/CSS/JS",
]


# ----------------------------- Data loading -------------------------------

def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TARGET_FILE}")

    inputs = pd.read_csv(INPUT_FILE)
    targets = pd.read_csv(TARGET_FILE)

    required_inputs = {"ID", "prefix", "type"}
    required_targets = {"commitID", "CPUTime"}

    if not required_inputs.issubset(inputs.columns):
        raise ValueError(f"InputData.csv columns: {inputs.columns.tolist()}")
    if not required_targets.issubset(targets.columns):
        raise ValueError(f"CPUTimes.csv columns: {targets.columns.tolist()}")
    if len(inputs) != 500 or len(targets) != 500:
        raise ValueError("Both supplied datasets must contain exactly 500 rows.")

    if not np.array_equal(targets["commitID"].to_numpy(), np.arange(1, 501)):
        raise ValueError("CPUTimes.csv commitID must be 1..500 for row-order alignment.")

    return pd.DataFrame({
        "prefix": inputs["prefix"].astype(str).to_numpy(),
        "type": inputs["type"].astype(str).to_numpy(),
        "CPUTime": targets["CPUTime"].astype(float).to_numpy(),
    })


# ----------------------------- Preprocessing ------------------------------

def one_hot(values: pd.Series, vocabulary: list[str]) -> np.ndarray:
    index = {value: i for i, value in enumerate(vocabulary)}
    encoded = np.zeros((len(values), len(vocabulary)), dtype=float)

    unknown = sorted(set(values.astype(str)) - set(vocabulary))
    if unknown:
        raise ValueError(f"Unknown categorical values: {unknown}")

    for row, value in enumerate(values.astype(str)):
        encoded[row, index[value]] = 1.0

    return encoded


def build_crossed_features(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Same prefix/type + prefix×type one-hot representation used in Part 1."""
    prefix_features = one_hot(data["prefix"], PREFIX_VOCAB)
    type_features = one_hot(data["type"], TYPE_VOCAB)

    names = [f"prefix={p}" for p in PREFIX_VOCAB]
    names += [f"type={t}" for t in TYPE_VOCAB]

    cross_vocab = [
        (prefix, file_type)
        for prefix in PREFIX_VOCAB
        for file_type in TYPE_VOCAB
    ]
    cross_values = list(zip(data["prefix"], data["type"]))

    cross_index = {value: i for i, value in enumerate(cross_vocab)}
    cross_features = np.zeros((len(data), len(cross_vocab)), dtype=float)

    for row, value in enumerate(cross_values):
        if value not in cross_index:
            raise ValueError(f"Unknown prefix/type combination: {value}")
        cross_features[row, cross_index[value]] = 1.0

    names += [f"prefix×type={p} || {t}" for p, t in cross_vocab]

    return np.hstack([prefix_features, type_features, cross_features]), names


# ----------------------------- Metrics ------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ----------------------------- Comparison ---------------------------------

def load_part1_final_test_results() -> pd.DataFrame:
    if not PART1_RESULTS_FILE.exists():
        raise FileNotFoundError(
            "part1_period_results.csv is missing. Run part1_reproduction.py first."
        )

    results = pd.read_csv(PART1_RESULTS_FILE)
    required = {"period", "test_RMSE", "experiment"}
    if not required.issubset(results.columns):
        raise ValueError(
            f"{PART1_RESULTS_FILE.name} is missing one of {sorted(required)}"
        )

    final_period = results[results["period"] == results["period"].max()].copy()
    return final_period[["experiment", "test_RMSE"]]


# ----------------------------- Main experiment ---------------------------

def main() -> None:
    data = load_dataset()
    X, feature_names = build_crossed_features(data)
    y = data["CPUTime"].to_numpy(dtype=float)

    # Same row-order split as Part 1.
    X_train = X[:N_TRAIN]
    y_train = y[:N_TRAIN]

    X_validation = X[N_TRAIN:N_TRAIN + N_VALIDATION]
    y_validation = y[N_TRAIN:N_TRAIN + N_VALIDATION]

    X_test = X[N_TRAIN + N_VALIDATION:]
    y_test = y[N_TRAIN + N_VALIDATION:]

    if len(X_test) != N_TEST:
        raise AssertionError("Unexpected test-set size.")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    validation_pred = model.predict(X_validation)
    test_pred = model.predict(X_test)

    train_rmse = rmse(y_train, train_pred)
    validation_rmse = rmse(y_validation, validation_pred)
    test_rmse = rmse(y_test, test_pred)

    predictions = pd.DataFrame({
        "row": np.arange(N_TRAIN + N_VALIDATION + 1, 501),
        "actual_CPUTime_ms": y_test,
        "predicted_CPUTime_ms": test_pred,
        "absolute_error_ms": np.abs(y_test - test_pred),
    })
    predictions.to_csv(DATA_DIR / "part2_random_forest_test_predictions.csv", index=False)

    results = pd.DataFrame([
        {"split": "training", "RMSE_ms": train_rmse},
        {"split": "validation", "RMSE_ms": validation_rmse},
        {"split": "testing", "RMSE_ms": test_rmse},
    ])
    results.to_csv(DATA_DIR / "part2_random_forest_results.csv", index=False)

    # Fair model comparison on the same TEST set.
    part1_final = load_part1_final_test_results()
    comparison_rows = []
    for _, row in part1_final.iterrows():
        comparison_rows.append({
            "model": row["experiment"],
            "test_RMSE_ms": float(row["test_RMSE"]),
        })
    comparison_rows.append({
        "model": "Random Forest Regressor",
        "test_RMSE_ms": test_rmse,
    })

    comparison = pd.DataFrame(comparison_rows)
    best_reference = float(comparison.iloc[:-1]["test_RMSE_ms"].min())

    comparison["difference_vs_Random_Forest_ms"] = (
        comparison["test_RMSE_ms"] - test_rmse
    )
    comparison["Random_Forest_relative_to_this_model"] = np.where(
        comparison["test_RMSE_ms"] > test_rmse,
        "Random Forest lower RMSE",
        np.where(
            comparison["test_RMSE_ms"] < test_rmse,
            "Random Forest higher RMSE",
            "Same RMSE",
        ),
    )
    comparison.to_csv(DATA_DIR / "part2_model_comparison.csv", index=False)

    # Feature importance plot for interpretation in the report.
    try:
        import matplotlib.pyplot as plt

        importances = pd.Series(model.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(15).sort_values()

        plt.figure(figsize=(9, 6))
        plt.barh(top.index, top.values)
        plt.xlabel("Random Forest feature importance")
        plt.ylabel("Feature")
        plt.title("Top Random Forest Feature Importances")
        plt.tight_layout()
        plt.savefig(DATA_DIR / "part2_random_forest_feature_importance.png", dpi=200)
        plt.close()
    except ImportError:
        print("matplotlib not installed; feature-importance plot was skipped.")

    print("=" * 72)
    print("PART 2 — RANDOM FOREST REGRESSOR")
    print("=" * 72)
    print(f"Dataset: 500 examples")
    print(f"Split: {N_TRAIN} train / {N_VALIDATION} validation / {N_TEST} test")
    print("Features: prefix + type + prefix×type one-hot representation")
    print(
        f"Hyperparameters: n_estimators={N_ESTIMATORS}, "
        f"max_features={MAX_FEATURES}, min_samples_leaf={MIN_SAMPLES_LEAF}, "
        f"random_state={RANDOM_STATE}"
    )
    print("\nRMSE:")
    print(f"  Training   : {train_rmse:.2f} ms")
    print(f"  Validation : {validation_rmse:.2f} ms")
    print(f"  Testing    : {test_rmse:.2f} ms")

    print("\nFair TEST-set comparison:")
    print(comparison[["model", "test_RMSE_ms"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nGenerated files:")
    print(f"  {DATA_DIR / 'part2_random_forest_results.csv'}")
    print(f"  {DATA_DIR / 'part2_random_forest_test_predictions.csv'}")
    print(f"  {DATA_DIR / 'part2_model_comparison.csv'}")
    print(f"  {DATA_DIR / 'part2_random_forest_feature_importance.png'}")


if __name__ == "__main__":
    main()

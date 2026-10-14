"""
Part 1 — Reproduction of:
"Predicting Bazel Build Times Using Machine Learning"

This implementation is based ONLY on the reference paper and the supplied
InputData.csv / CPUTimes.csv files. It intentionally does not import, copy,
or depend on the authors' original Python implementation.

Reference details taken from the paper:
- 500 Bazel commits
- input features: most common file-path prefix and most common file type
- target: CPU time (milliseconds)
- categorical vocabularies are listed in the paper
- 300 training examples, 100 validation examples, 100 test examples
- Linear Regression trained using stochastic gradient descent
- learning rate = 0.01
- batch size = 16
- the results tables show 9 training periods
- evaluation metric = RMSE
- a second experiment adds a prefix × type feature crossing

Reproduction assumptions that are necessary because the paper does not specify
them completely:
- The two supplied CSVs have no shared commit identifier. Their rows are
  therefore aligned by order: InputData row 1 <-> CPUTimes commitID 1, etc.
- The paper does not state a random train/validation/test split or a seed.
  We therefore use the supplied row order: 300 / 100 / 100, with no shuffle.
- The paper says it used 3-fold grid search, but it does not publish the full
  candidate grid. We therefore use the reported selected hyperparameters
  (learning rate 0.01, batch size 16) directly rather than inventing a grid.
- The paper's result tables contain 9 periods, so this implementation makes
  9 passes over the training set and records RMSE after each pass.
- Categorical inputs are represented as one-hot vectors. The crossed feature
  is a categorical pair (prefix, type), represented as another one-hot block.

The goal is to reproduce the documented methodology, not the authors' code.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------- Configuration -----------------------------

DATA_DIR = Path(__file__).resolve().parent
INPUT_FILE = DATA_DIR / "InputData.csv"
TARGET_FILE = DATA_DIR / "CPUTimes.csv"

LEARNING_RATE = 0.01
BATCH_SIZE = 16
EPOCHS = 9

N_TRAIN = 300
N_VALIDATION = 100
N_TEST = 100

# Vocabularies explicitly listed in the paper.
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

REPORTED_RMSE_NO_CROSS = 5735.84
REPORTED_RMSE_WITH_CROSS = 5628.09


# ----------------------------- Data loading -------------------------------

def load_dataset() -> pd.DataFrame:
    """Load and align the supplied input and target files."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TARGET_FILE}")

    inputs = pd.read_csv(INPUT_FILE)
    targets = pd.read_csv(TARGET_FILE)

    required_input_cols = {"ID", "prefix", "type"}
    required_target_cols = {"commitID", "CPUTime"}

    if not required_input_cols.issubset(inputs.columns):
        raise ValueError(
            f"InputData.csv must contain {sorted(required_input_cols)}; "
            f"found {inputs.columns.tolist()}"
        )

    if not required_target_cols.issubset(targets.columns):
        raise ValueError(
            f"CPUTimes.csv must contain {sorted(required_target_cols)}; "
            f"found {targets.columns.tolist()}"
        )

    if len(inputs) != 500 or len(targets) != 500:
        raise ValueError(
            "The supplied dataset is expected to contain exactly 500 rows "
            f"(got {len(inputs)} input rows and {len(targets)} target rows)."
        )

    if inputs[["prefix", "type"]].isna().any().any():
        raise ValueError("Input data contains missing prefix/type values.")
    if targets["CPUTime"].isna().any():
        raise ValueError("Target data contains missing CPUTime values.")

    # The two supplied files expose different identifiers.  The paper says
    # the input and output were recorded for each commit; in the supplied
    # processed files, CPUTime is numbered 1..500, so row-order alignment is
    # the reproducible interpretation available from the supplied data.
    if not np.array_equal(targets["commitID"].to_numpy(), np.arange(1, 501)):
        raise ValueError(
            "CPUTimes.csv commitID is not the expected 1..500 sequence; "
            "row-order alignment would no longer be justified."
        )

    data = pd.DataFrame({
        "prefix": inputs["prefix"].astype(str).to_numpy(),
        "type": inputs["type"].astype(str).to_numpy(),
        "CPUTime": targets["CPUTime"].astype(float).to_numpy(),
    })

    return data


# ----------------------------- Encoding -----------------------------------

def one_hot(values: pd.Series, vocabulary: list[str]) -> np.ndarray:
    """One-hot encode a categorical Series using the paper's fixed vocabulary."""
    index = {value: i for i, value in enumerate(vocabulary)}
    encoded = np.zeros((len(values), len(vocabulary)), dtype=float)

    unknown = sorted(set(values.astype(str)) - set(vocabulary))
    if unknown:
        raise ValueError(
            f"Found categories not listed in the paper's vocabulary: {unknown}"
        )

    for row, value in enumerate(values.astype(str)):
        encoded[row, index[value]] = 1.0

    return encoded


def build_features(data: pd.DataFrame, use_cross: bool) -> np.ndarray:
    """
    Build the categorical input vector.

    Without crossing:
        [one-hot(prefix), one-hot(type)]

    With crossing:
        [one-hot(prefix), one-hot(type), one-hot(prefix × type)]
    """
    prefix_features = one_hot(data["prefix"], PREFIX_VOCAB)
    type_features = one_hot(data["type"], TYPE_VOCAB)

    if not use_cross:
        return np.hstack([prefix_features, type_features])

    cross_vocab = [
        f"{prefix} || {file_type}"
        for prefix in PREFIX_VOCAB
        for file_type in TYPE_VOCAB
    ]
    cross_values = [
        f"{prefix} || {file_type}"
        for prefix, file_type in zip(data["prefix"], data["type"])
    ]
    cross_features = one_hot(pd.Series(cross_values), cross_vocab)

    return np.hstack([prefix_features, type_features, cross_features])


# ----------------------------- Model --------------------------------------

class MiniBatchLinearRegressionSGD:
    """
    Linear regression trained with mini-batch stochastic gradient descent.

    The update corresponds to minimizing 1/(2m) * sum((prediction - y)^2)
    on each mini-batch.
    """

    def __init__(self, learning_rate: float, batch_size: int):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit_one_epoch(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.weights is None:
            self.weights = np.zeros(X.shape[1], dtype=float)
            self.bias = 0.0

        # The paper does not document shuffling, so preserve supplied order.
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            y_batch = y[start:start + self.batch_size]

            predictions = X_batch @ self.weights + self.bias
            errors = predictions - y_batch

            grad_w = (X_batch.T @ errors) / len(X_batch)
            grad_b = float(np.mean(errors))

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model must be trained before prediction.")
        return X @ self.weights + self.bias


# ----------------------------- Metrics ------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ----------------------------- Experiment ---------------------------------

def run_experiment(data: pd.DataFrame, use_cross: bool) -> tuple[pd.DataFrame, float]:
    """Run the documented 9-period experiment and return per-period metrics."""
    X = build_features(data, use_cross=use_cross)
    y = data["CPUTime"].to_numpy(dtype=float)

    if len(X) != 500:
        raise AssertionError("Feature matrix must contain 500 examples.")

    X_train = X[:N_TRAIN]
    y_train = y[:N_TRAIN]

    X_val = X[N_TRAIN:N_TRAIN + N_VALIDATION]
    y_val = y[N_TRAIN:N_TRAIN + N_VALIDATION]

    X_test = X[N_TRAIN + N_VALIDATION:]
    y_test = y[N_TRAIN + N_VALIDATION:]

    if len(X_test) != N_TEST:
        raise AssertionError("Unexpected test-set size.")

    model = MiniBatchLinearRegressionSGD(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )

    rows = []

    for period in range(1, EPOCHS + 1):
        model.fit_one_epoch(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        rows.append({
            "period": period,
            "train_RMSE": rmse(y_train, train_pred),
            "validation_RMSE": rmse(y_val, val_pred),
            "test_RMSE": rmse(y_test, test_pred),
        })

    history = pd.DataFrame(rows)
    average_train_rmse = float(history["train_RMSE"].mean())

    return history, average_train_rmse


# ----------------------------- Reporting ----------------------------------

def print_experiment_summary(
    label: str,
    history: pd.DataFrame,
    reported_rmse: float,
) -> None:
    average_train = history["train_RMSE"].mean()
    average_val = history["validation_RMSE"].mean()
    average_test = history["test_RMSE"].mean()

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(history.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nAverage RMSE across 9 periods:")
    print(f"  Training   : {average_train:.2f} ms")
    print(f"  Validation : {average_val:.2f} ms")
    print(f"  Testing    : {average_test:.2f} ms")

    difference = average_train - reported_rmse
    percent = (difference / reported_rmse) * 100.0

    print("\nComparison with paper:")
    print(f"  Reported training RMSE: {reported_rmse:.2f} ms")
    print(f"  Your training RMSE    : {average_train:.2f} ms")
    print(f"  Difference            : {difference:+.2f} ms ({percent:+.2f}%)")


def save_outputs(
    no_cross_history: pd.DataFrame,
    cross_history: pd.DataFrame,
) -> None:
    no_cross_history = no_cross_history.copy()
    cross_history = cross_history.copy()

    no_cross_history["experiment"] = "Linear Regression - without feature crossing"
    cross_history["experiment"] = "Linear Regression - with prefix × type feature crossing"

    combined = pd.concat(
        [no_cross_history, cross_history],
        ignore_index=True,
    )

    results_file = DATA_DIR / "part1_period_results.csv"
    combined.to_csv(results_file, index=False)

    summary = pd.DataFrame([
        {
            "experiment": "Linear Regression - without feature crossing",
            "reported_training_RMSE_ms": REPORTED_RMSE_NO_CROSS,
            "your_training_RMSE_ms": no_cross_history["train_RMSE"].mean(),
            "your_validation_RMSE_ms": no_cross_history["validation_RMSE"].mean(),
            "your_test_RMSE_ms": no_cross_history["test_RMSE"].mean(),
        },
        {
            "experiment": "Linear Regression - with prefix × type feature crossing",
            "reported_training_RMSE_ms": REPORTED_RMSE_WITH_CROSS,
            "your_training_RMSE_ms": cross_history["train_RMSE"].mean(),
            "your_validation_RMSE_ms": cross_history["validation_RMSE"].mean(),
            "your_test_RMSE_ms": cross_history["test_RMSE"].mean(),
        },
    ])

    summary_file = DATA_DIR / "part1_reported_vs_yours.csv"
    summary.to_csv(summary_file, index=False)

    # Separate plots, matching the paper's training/validation presentation.
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(
            no_cross_history["period"],
            no_cross_history["train_RMSE"],
            marker="o",
            label="Without feature crossing",
        )
        plt.plot(
            cross_history["period"],
            cross_history["train_RMSE"],
            marker="o",
            label="With feature crossing",
        )
        plt.xlabel("Period")
        plt.ylabel("Training RMSE (ms)")
        plt.title("Part 1: Training RMSE by Period")
        plt.legend()
        plt.tight_layout()
        plt.savefig(DATA_DIR / "part1_training_rmse.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(
            no_cross_history["period"],
            no_cross_history["validation_RMSE"],
            marker="o",
            label="Without feature crossing",
        )
        plt.plot(
            cross_history["period"],
            cross_history["validation_RMSE"],
            marker="o",
            label="With feature crossing",
        )
        plt.xlabel("Period")
        plt.ylabel("Validation RMSE (ms)")
        plt.title("Part 1: Validation RMSE by Period")
        plt.legend()
        plt.tight_layout()
        plt.savefig(DATA_DIR / "part1_validation_rmse.png", dpi=200)
        plt.close()
    except ImportError:
        print("\nmatplotlib not installed; CSV outputs were still created.")

    print("\nSaved:")
    print(f"  {results_file}")
    print(f"  {summary_file}")
    print(f"  {DATA_DIR / 'part1_training_rmse.png'}")
    print(f"  {DATA_DIR / 'part1_validation_rmse.png'}")


# ----------------------------- Main ----------------------------------------

def main() -> None:
    data = load_dataset()

    print("Loaded dataset successfully.")
    print(f"Rows: {len(data)}")
    print(f"Unique prefixes in supplied data: {sorted(data['prefix'].unique())}")
    print(f"Unique types in supplied data: {sorted(data['type'].unique())}")
    print(
        f"Split: {N_TRAIN} train / {N_VALIDATION} validation / "
        f"{N_TEST} test"
    )
    print(
        f"Hyperparameters used: learning_rate={LEARNING_RATE}, "
        f"batch_size={BATCH_SIZE}, periods={EPOCHS}"
    )

    no_cross_history, no_cross_avg = run_experiment(data, use_cross=False)
    cross_history, cross_avg = run_experiment(data, use_cross=True)

    print_experiment_summary(
        "LINEAR REGRESSION WITHOUT FEATURE CROSSING",
        no_cross_history,
        REPORTED_RMSE_NO_CROSS,
    )

    print_experiment_summary(
        "LINEAR REGRESSION WITH PREFIX × TYPE FEATURE CROSSING",
        cross_history,
        REPORTED_RMSE_WITH_CROSS,
    )

    save_outputs(no_cross_history, cross_history)


if __name__ == "__main__":
    main()

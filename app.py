import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "InputData.csv"
TARGET_FILE = BASE_DIR / "CPUTimes.csv"

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

# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_dataset():
    inputs = pd.read_csv(INPUT_FILE)
    targets = pd.read_csv(TARGET_FILE)

    if len(inputs) != 500 or len(targets) != 500:
        raise ValueError("Expected exactly 500 input and target records.")

    return pd.DataFrame({
        "prefix": inputs["prefix"].astype(str),
        "type": inputs["type"].astype(str),
        "CPUTime": targets["CPUTime"].astype(float),
    })


# --------------------------------------------------
# Build the same categorical features
# used in the Random Forest experiment
# --------------------------------------------------

def build_features(data):
    prefix_index = {value: i for i, value in enumerate(PREFIX_VOCAB)}
    type_index = {value: i for i, value in enumerate(TYPE_VOCAB)}

    cross_vocab = [
        (prefix, file_type)
        for prefix in PREFIX_VOCAB
        for file_type in TYPE_VOCAB
    ]
    cross_index = {value: i for i, value in enumerate(cross_vocab)}

    features = np.zeros(
        (len(data), len(PREFIX_VOCAB) + len(TYPE_VOCAB) + len(cross_vocab))
    )

    for row, (prefix, file_type) in enumerate(
        zip(data["prefix"], data["type"])
    ):
        if prefix not in prefix_index or file_type not in type_index:
            raise ValueError("An input category is outside the known vocabulary.")

        features[row, prefix_index[prefix]] = 1
        features[row, len(PREFIX_VOCAB) + type_index[file_type]] = 1

        cross_position = (
            len(PREFIX_VOCAB)
            + len(TYPE_VOCAB)
            + cross_index[(prefix, file_type)]
        )
        features[row, cross_position] = 1

    return features


# --------------------------------------------------
# Train the model
# --------------------------------------------------

@st.cache_resource
def train_model():
    data = load_dataset()
    X = build_features(data)
    y = data["CPUTime"].to_numpy()

    # Same 300-row training split as the submitted experiment.
    model = RandomForestRegressor(
        n_estimators=300,
        max_features=1.0,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X[:300], y[:300])

    return model


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="Bazel Build-Time Predictor",
    page_icon="⚙️",
    layout="centered",
)

st.title("⚙️ Bazel Build-Time Predictor")

st.write(
    "Estimate Bazel build CPU time from the most common "
    "file-path prefix and file type."
)

st.info(
    "This is an experimental machine-learning estimate, "
    "not a guaranteed build duration."
)

try:
    data = load_dataset()
    model = train_model()

    st.success("Dataset loaded and Random Forest model trained.")

    prefix = st.selectbox(
        "Select file-path prefix",
        options=sorted(data["prefix"].unique()),
    )

    file_type = st.selectbox(
        "Select file type",
        options=sorted(data["type"].unique()),
    )

    if st.button("Predict Build Time", type="primary", use_container_width=True):
        sample = pd.DataFrame({
            "prefix": [prefix],
            "type": [file_type],
        })

        prediction = float(model.predict(build_features(sample))[0])

        st.subheader("Prediction")
        st.metric("Estimated CPU time", f"{prediction:,.2f} ms")
        st.caption(f"Approximately {prediction / 1000:.2f} seconds.")

        st.warning(
            "The prediction is based on the supplied dataset and "
            "the selected categorical features. Actual build times "
            "can differ."
        )

    with st.expander("About this model"):
        st.write("Model: Random Forest Regressor")
        st.write("Training records: 300")
        st.write("Trees: 300")
        st.write("Random seed: 42")
        st.write("Features: prefix, type, and prefix × type interaction")
        st.write("Target: CPUTime in milliseconds")

except Exception as error:
    st.error(f"The predictor could not start: {error}")
    st.info(
        "Check that InputData.csv and CPUTimes.csv are in the "
        "same folder as app.py."
    )
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

import os

BASE_DIR = os.path.dirname(__file__)

# model = joblib.load(
#     os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
# )

# importance = pd.read_csv(
#     os.path.join(BASE_DIR, "..", "models", "feature_importance.csv")
# )
@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
    )

@st.cache_data
def load_importance():
    return pd.read_csv(
        os.path.join(BASE_DIR, "..", "models", "feature_importance.csv")
    )

model = load_model()
importance = load_importance()

# =====================================
# HEADER
# =====================================

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
Detect potentially fraudulent transactions using a Machine Learning model trained on **284K+ credit card transactions**.

This application uses a **Random Forest + SMOTE** model to identify suspicious transactions.
""")
# =====================================
# ANALYSIS MODE
# =====================================

mode = st.radio(
    "Select Analysis Type",
    [
        "Single Transaction",
        "Bulk CSV Analysis"
    ]
)

st.markdown("---")

# =====================================
# TOP METRICS
# =====================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Recall",
        "82.7%"
    )

with metric2:
    st.metric(
        "Precision",
        "84%"
    )

with metric3:
    st.metric(
        "Dataset",
        "284K+"
    )

st.markdown("---")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📊 Project Information")

st.sidebar.success(
    "Random Forest + SMOTE"
)

st.sidebar.write("""
### Dataset

- 284,807 Transactions
- 492 Fraud Cases
- Highly Imbalanced Dataset
""")

st.sidebar.write("""
### Top Features

- V14
- V10
- V4
- V12
- V17
""")

st.sidebar.write("""
### Technologies Used

- Python
- Scikit-Learn
- Random Forest
- SMOTE
- Streamlit
""")
if mode == "Single Transaction":

    # =====================================
    # INPUT SECTION
    # =====================================

    st.header("📝 Transaction Details")

    left, right = st.columns(2)

    with left:

        V14 = st.number_input(
            "V14",
            value=0.0,
            format="%.4f"
        )

        V10 = st.number_input(
            "V10",
            value=0.0,
            format="%.4f"
        )

        V4 = st.number_input(
            "V4",
            value=0.0,
            format="%.4f"
        )

    with right:

        V12 = st.number_input(
            "V12",
            value=0.0,
            format="%.4f"
        )

        V17 = st.number_input(
            "V17",
            value=0.0,
            format="%.4f"
        )

    predict = st.button(
        "🚀 Analyze Transaction",
        use_container_width=True
    )

    if predict:
        features = np.zeros((1, 30))
        features[0][14] = V14
        features[0][10] = V10
        features[0][4] = V4
        features[0][12] = V12
        features[0][17] = V17
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]
        risk_score = probability * 100
        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        if prediction[0] == 1:
            st.error("🚨 Fraudulent Transaction Detected")
        else:
            st.success("✅ Transaction is Genuine")    
        st.metric(
            "Fraud Risk Score",
            f"{risk_score:.2f}%"
        )   
        st.progress(float(probability))

        if risk_score < 30:
            st.success("🟢 Low Risk Transaction")
        elif risk_score < 70:
            st.warning("🟡 Medium Risk Transaction")
        else:
            st.error("🔴 High Risk Transaction")


# =====================================
# BULK CSV ANALYSIS
# =====================================

if mode == "Bulk CSV Analysis":

    st.header("📂 Bulk CSV Fraud Detection")

    st.info(
        """
        Upload a CSV file containing
        all 30 model features (V1-V28, Time, Amount)
        in the same format used during training.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.subheader("📄 Uploaded Data")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            if st.button(
                "🚀 Analyze CSV"
            ):

                predictions = model.predict(df)

                probabilities = (
                    model
                    .predict_proba(df)[:, 1]
                )

                results = df.copy()

                results[
                    "Fraud Prediction"
                ] = predictions

                results[
                    "Fraud Probability"
                ] = probabilities

                fraud_count = (
                    predictions == 1
                ).sum()

                total_count = len(results)

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Total Transactions",
                        total_count
                    )

                with c2:
                    st.metric(
                        "Fraud Transactions",
                        fraud_count
                    )

                with c3:
                    st.metric(
                        "Fraud Rate",
                        f"{fraud_count/total_count*100:.2f}%"
                    )

                st.markdown("---")

                st.subheader(
                    "🚨 High Risk Transactions"
                )

                suspicious = (
                    results
                    .sort_values(
                        by="Fraud Probability",
                        ascending=False
                    )
                    .head(20)
                )

                st.dataframe(
                    suspicious,
                    use_container_width=True
                )

                csv = (
                    results
                    .to_csv(index=False)
                    .encode("utf-8")
                )

                st.download_button(
                    "⬇️ Download Results",
                    data=csv,
                    file_name=
                    "fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )
# =====================================
# MODEL COMPARISON
# =====================================

st.markdown("---")

st.subheader("📊 Model Comparison")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Random Forest + SMOTE"
    ],
    "Recall (%)": [
        69.4,
        79.6,
        78.6,
        82.7
    ]
})

st.dataframe(
    comparison,
    use_container_width=True
)

# =====================================
# FEATURE IMPORTANCE
# =====================================

st.markdown("---")

st.subheader(
    "📈 Top 10 Important Features"
)

chart_data = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .head(10)
)

st.bar_chart(
    chart_data.set_index("Feature")
)

# =====================================
# EXPLAINABILITY
# =====================================

st.markdown("---")

with st.expander(
    "🔍 How does the model work?"
):

    st.write("""
This model was trained using:

- Random Forest Classifier
- SMOTE Oversampling
- 284K+ credit card transactions

Most influential features:

- V14
- V10
- V4
- V12
- V17

Fraudulent transactions generally exhibit abnormal values in these features.
""")

# =====================================
# PROJECT SUMMARY
# =====================================

st.markdown("---")

st.subheader("📌 Project Summary")

st.info("""
This project compares multiple Machine Learning models for fraud detection and uses SMOTE to address class imbalance.

Final Model:
Random Forest + SMOTE

Performance:
- Recall: 82.7%
- Precision: 84%
- F1 Score: 83%
""")

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
### 👨‍💻 Developer

**Akriti Gupta**  
B.Tech, IIT BHU

Machine Learning | Data Science | AI
""")
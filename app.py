import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("loan_bundle.pkl")


b = load_bundle()
ohe = b["ohe"]
cat_cols = b["ohe_cols"]
cat_options = {c: list(cats) for c, cats in zip(cat_cols, ohe.categories_)}
edu_map = b["education_map"]

st.title("🏦 CreditWise Loan Approval System")
st.caption("SecureTrust Bank | ML-powered decision support for loan approval")

# ---------------- Sidebar ----------------
st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Choose model", list(b["models"].keys()), index=0)
st.sidebar.info(
    "Predictions are decision support only. Final approval requires human verification."
)

# ---------------- Form ----------------
st.subheader("Applicant Details")
c1, c2, c3 = st.columns(3)

with c1:
    applicant_income = st.number_input("Applicant Income", 0.0, 1_000_000.0, 10000.0, step=500.0)
    coapplicant_income = st.number_input("Coapplicant Income", 0.0, 1_000_000.0, 5000.0, step=500.0)
    age = st.number_input("Age", 18, 80, 35)
    dependents = st.number_input("Dependents", 0, 10, 1)
    credit_score = st.number_input("Credit Score", 300, 900, 680)
    existing_loans = st.number_input("Existing Loans", 0, 20, 1)

with c2:
    dti_ratio = st.number_input("DTI Ratio", 0.0, 1.0, 0.35, step=0.01)
    savings = st.number_input("Savings", 0.0, 10_000_000.0, 10000.0, step=500.0)
    collateral_value = st.number_input("Collateral Value", 0.0, 10_000_000.0, 25000.0, step=500.0)
    loan_amount = st.number_input("Loan Amount", 0.0, 10_000_000.0, 20000.0, step=500.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84], index=3)
    education = st.selectbox("Education Level", list(edu_map.keys()))

with c3:
    cat_inputs = {}
    for col in cat_cols:
        cat_inputs[col] = st.selectbox(col.replace("_", " "), cat_options[col])

# ---------------- Predict ----------------
if st.button("Predict Loan Decision", type="primary"):
    num_row = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": edu_map[education],
    }])

    # One-hot encode categorical columns exactly as in training
    enc = ohe.transform(pd.DataFrame([cat_inputs])[cat_cols])
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(cat_cols))
    full = pd.concat([num_row, enc_df], axis=1)

    # Feature engineering exactly as in training
    full["DTI_Ratio_sq"] = full["DTI_Ratio"] ** 2
    full["Credit_Score_sq"] = full["Credit_Score"] ** 2
    full = full.drop(columns=["Credit_Score", "DTI_Ratio"])

    # Same column order as training, then scale
    full = full.reindex(columns=b["feature_names"], fill_value=0)
    scaled = b["scaler"].transform(full)

    model = b["models"][model_name]
    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]

    st.divider()
    if pred == 1:
        st.success(f"✅ Loan APPROVED  (confidence: {proba[1]:.0%})")
    else:
        st.error(f"❌ Loan REJECTED  (confidence: {proba[0]:.0%})")

    st.progress(float(proba[1]), text=f"Approval probability: {proba[1]:.0%}")
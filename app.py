import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Stellar Object Classifier", page_icon="🔭")

FEATURES = ["u", "g", "r", "i", "z", "redshift"]
LABELS = ["STAR", "GALAXY", "QSO (Quasar)"]

PRESETS = {  # illustrative typical values, not real catalogue entries
    "Typical star":   dict(u=19.5, g=17.8, r=17.2, i=16.9, z=16.7, redshift=0.0001),
    "Typical galaxy": dict(u=21.5, g=19.8, r=18.7, i=18.2, z=17.8, redshift=0.12),
    "Typical quasar": dict(u=19.8, g=19.5, r=19.2, i=19.0, z=18.8, redshift=1.8),
}


@st.cache_resource
def load_model():
    m = XGBClassifier()
    m.load_model("model.json")
    return m


model = load_model()

for k, v in PRESETS["Typical galaxy"].items():
    st.session_state.setdefault(k, v)


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[k] = v


st.title("🔭 Stellar Object Classifier")
st.write(
    "Classifies SDSS objects as **Star, Galaxy or Quasar** from five photometric "
    "magnitudes and spectroscopic redshift. Trained on SDSS DR17 (100,000 objects)."
)

st.selectbox("Load an example", list(PRESETS), key="preset", on_change=apply_preset)

c1, c2 = st.columns(2)
with c1:
    st.number_input("u (ultraviolet)", key="u", format="%.3f", step=0.1)
    st.number_input("g (green)", key="g", format="%.3f", step=0.1)
    st.number_input("r (red)", key="r", format="%.3f", step=0.1)
with c2:
    st.number_input("i (near-infrared)", key="i", format="%.3f", step=0.1)
    st.number_input("z (infrared)", key="z", format="%.3f", step=0.1)
    st.number_input("redshift", key="redshift", format="%.5f", step=0.01)

if st.button("Classify", type="primary"):
    x = pd.DataFrame([[st.session_state[f] for f in FEATURES]], columns=FEATURES)
    probs = model.predict_proba(x)[0]
    best = int(np.argmax(probs))
    st.success(f"Prediction: **{LABELS[best]}** ({probs[best]:.1%} confidence)")
    st.bar_chart(pd.DataFrame({"Probability": probs}, index=LABELS))

st.caption("Data: Sloan Digital Sky Survey DR17 via Kaggle. Model: XGBoost.")
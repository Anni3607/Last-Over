import streamlit as st
import joblib

# Load model and encoders
model = joblib.load('chase_predictor.pkl')
le_batsman = joblib.load('le_batsman.pkl')
le_bowler = joblib.load('le_bowler.pkl')

# Get known names from encoders
batsmen_list = list(le_batsman.classes_)
bowlers_list = list(le_bowler.classes_)

st.set_page_config(page_title="🏏 Last Over Chase Predictor", layout="centered")

st.title("🏏 Last Over Chase Predictor")
st.markdown("Enter the **batsman** and **bowler** to predict whether **10 runs will be chased in the last over**.")

# Dropdowns instead of text input
batsman = st.selectbox("🧤 Select Batsman", batsmen_list)
bowler = st.selectbox("🔥 Select Bowler", bowlers_list)

if st.button("Predict"):
    batsman_enc = le_batsman.transform([batsman])[0]
    bowler_enc = le_bowler.transform([bowler])[0]
    
    pred = model.predict([[batsman_enc, bowler_enc]])[0]
    
    if pred == 1:
        st.success("🎉 yaayy! Likely to chase 15 runs!")
    else:
        st.error("❌ nayyy! Unlikely to chase 15 runs.")

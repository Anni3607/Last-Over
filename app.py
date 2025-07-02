
import streamlit as st
import joblib

model = joblib.load('chase_predictor.pkl')
le_batsman = joblib.load('le_batsman.pkl')
le_bowler = joblib.load('le_bowler.pkl')

st.set_page_config(page_title="🏏 Last Over Chase Predictor", layout="centered")

st.title("🏏 Last Over Chase Predictor (15 Runs)")
st.markdown("Enter the **batsman** and **bowler** to predict whether 12 runs will be chased in the last over.")

batsman = st.text_input("🧤 Batsman Name (e.g. MS Dhoni)")
bowler = st.text_input("🔥 Bowler Name (e.g. Jasprit Bumrah)")

if st.button("Predict"):
    try:
        batsman_enc = le_batsman.transform([batsman])[0]
        bowler_enc = le_bowler.transform([bowler])[0]
        pred = model.predict([[batsman_enc, bowler_enc]])[0]
        if pred == 1:
            st.success("🎉 yaayy! Likely to chase 15 runs!")
        else:
            st.error("❌ nayyy! Unlikely to chase 15 runs.")
    except:
        st.warning("⚠️ Batsman or Bowler not in training data. Try known names.")

import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import sqlite3
import time
from groq import Groq

# -------------------- CONFIG --------------------
st.set_page_config(page_title="MedAI Assist PRO", layout="wide")

# -------------------- GROQ --------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# -------------------- DATABASE --------------------
conn = sqlite3.connect("data.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS history
             (symptoms TEXT, disease TEXT, probability REAL)''')

# -------------------- LOAD MODEL --------------------
try:
    model = pickle.load(open("model/model.pkl", "rb"))
except:
    st.error("❌ Model not found!")

try:
    with open("model/symptoms_list.json") as f:
        symptoms_list = json.load(f)
except:
    st.error("❌ Symptoms list not found!")

# -------------------- SESSION --------------------
if "results" not in st.session_state:
    st.session_state.results = None

if "symptoms" not in st.session_state:
    st.session_state.symptoms = []

# -------------------- HEADER --------------------
st.markdown("""
<div style="text-align:center;">
    <img src="https://cdn-icons-png.flaticon.com/512/4320/4320371.png" width="80">
    <h1>MedAI Assist PRO</h1>
    <p style="color:gray;">AI Powered Medical Assistant</p>
</div>
""", unsafe_allow_html=True)
# -------------------- SIDEBAR --------------------
menu = st.sidebar.selectbox("📌 Navigation", ["🏠 Home", "ℹ️ About", "📜 History"])
# ================= HOME =================
if menu == "🏠 Home":

    col1, col2 = st.columns(2)

    # ================= LEFT =================
    with col1:
        st.subheader("🩺 Select Symptoms")

        selected_symptoms = st.multiselect("Choose symptoms", symptoms_list)

        if st.button("🔍 Analyze Symptoms"):

            if not selected_symptoms:
                st.warning("⚠ Select symptoms first!")
            else:
                with st.spinner("Analyzing..."):
                    time.sleep(1)

                input_data = [1 if s in selected_symptoms else 0 for s in symptoms_list]
                input_data = np.array(input_data).reshape(1, -1)

                prediction = model.predict_proba(input_data)[0]

                results = sorted(
                    zip(model.classes_, prediction),
                    key=lambda x: x[1],
                    reverse=True
                )

                # SAVE
                st.session_state.results = results
                st.session_state.symptoms = selected_symptoms

                # SAVE HISTORY
                c.execute("INSERT INTO history VALUES (?, ?, ?)", (
                    ", ".join(selected_symptoms),
                    results[0][0],
                    float(results[0][1])
                ))
                conn.commit()

    # ================= RIGHT =================
    with col2:
        st.subheader("📊 AI Diagnosis Report")

        if st.session_state.results:

            results = st.session_state.results
            top_results = results[:5]

            st.markdown("### 🔝 Top Predictions")

            for disease, prob in top_results:
                percent = round(prob * 100, 2)

                st.markdown(f"""
                <div style="background:#1f1f1f;padding:12px;border-radius:10px;margin-bottom:10px;">
                    <b>{disease}</b><br>
                    Confidence: {percent}%
                </div>
                """, unsafe_allow_html=True)

                st.progress(prob)

            # Risk
            top_prob = top_results[0][1]

            if top_prob > 0.7:
                st.error("🚨 High Risk")
            elif top_prob > 0.4:
                st.warning("⚠ Medium Risk")
            else:
                st.success("✅ Low Risk")

            # AI RESULT
            try:
                with st.spinner("🧠 Generating AI Advice..."):

                    ai_response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a medical assistant. Give short and clear answers."
                            },
                            {
                                "role": "user",
                                "content": f"""
Symptoms: {st.session_state.symptoms}
Diseases: {[d[0] for d in top_results]}

Give:
- Reason
- Precautions
- Medicines
- Doctor

Short answer only.
"""
                            }
                        ]
                    )

                    st.info(ai_response.choices[0].message.content)

            except:
                st.error("⚠ AI Error")

        else:
            st.info("👉 Select symptoms and click Analyze")

elif menu == "📜 History":
    st.subheader("📜 Prediction History")

    data = c.execute("SELECT * FROM history").fetchall()

    if data:
        df = pd.DataFrame(data, columns=["Symptoms", "Disease", "Confidence"])
        st.dataframe(df)
    else:
        st.info("No history yet")

elif menu == "ℹ️ About":
    st.subheader("ℹ️ About MedAI Assist PRO")

    st.markdown("""
    ### 🧠 AI Medical Assistant

    This app predicts diseases using:
    - Machine Learning Model
    - Groq AI

    ### ⚙️ Features
    - Disease Prediction
    - AI Advice
    - Chat Assistant
    - History Tracking

    ### 👨‍💻 Developer
    Nishant Sonker
    """)

# ================= CHAT =================
st.markdown("---")
st.subheader("💬 AI Doctor Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask your question...")

colA, colB = st.columns([1,1])

with colA:
    send = st.button("Send 🚀")

with colB:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Show chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div style='text-align:center;background:#333;padding:10px;border-radius:15px;margin:5px;'>🧑 {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background:#111;padding:10px;border-radius:15px;margin:5px;'>🤖 {msg['content']}</div>", unsafe_allow_html=True)

# Send logic
if send and user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=st.session_state.messages
        )

        reply = response.choices[0].message.content

    except:
        reply = "⚠ API Error"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<center>© 2026 MedAI Assist PRO | Developed by Nishant Sonker </center>", unsafe_allow_html=True)
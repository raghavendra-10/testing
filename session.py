# streamlit_enhanced_issue_tracker.py
import streamlit as st
import pandas as pd
import json
import os
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import openai

# ----------- CONFIG -----------
DATA_FILE = "session_data.json"
model = SentenceTransformer("all-MiniLM-L6-v2")
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as environment variable

# ----------- UTILS -----------
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def extract_issues(session_json):
    return list(set(session_json.get("issues", [])))

# ----------- GPT HELPERS -----------
def get_tone_score(prev_issue, current_issue):
    prompt = f"Compare these two statements and tell if the tone improved, worsened, or stayed the same.\\n\\nPrevious: {prev_issue}\\nCurrent: {current_issue}\\n\\nReply with only one word: 'improved', 'same', or 'worsened'."
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        reply = res.choices[0].message.content.strip().lower()
        return 1 if reply == "improved" else 0
    except Exception:
        return 0

def get_emotional_intensity(issue_text):
    prompt = f"Rate the emotional intensity of this statement from 1 (very mild) to 5 (very intense): '{issue_text}'"
    try:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        score = int([int(s) for s in res.choices[0].message.content if s.isdigit()][0])
        return score
    except Exception:
        return 3

# ----------- SCORING -----------
def score_issues_semantic(session_issues_list):
    issue_history = {}
    embeddings = []

    for i, issues in enumerate(session_issues_list):
        vecs = model.encode(issues)
        embeddings.append((issues, vecs))

        if i == 0:
            for issue in issues:
                issue_history[issue] = {"sessions": [i], "duration": 1, "intensity": get_emotional_intensity(issue)}
        else:
            prev_issues, prev_vecs = embeddings[i - 1]
            matched = set()

            for j, vec in enumerate(vecs):
                best_score = -1
                best_match = None

                for k, prev_vec in enumerate(prev_vecs):
                    sim = util.cos_sim(vec, prev_vec).item()
                    if sim > 0.8 and sim > best_score:
                        best_score = sim
                        best_match = prev_issues[k]

                if best_match:
                    tone = get_tone_score(best_match, issues[j])
                    issue_history[best_match]["sessions"].append(i)
                    issue_history[best_match]["duration"] += 1
                    issue_history[best_match]["tone_score"] = tone
                else:
                    issue_history[issues[j]] = {"sessions": [i], "duration": 1, "intensity": get_emotional_intensity(issues[j])}

    latest_session = len(session_issues_list) - 1
    results = []
    for issue, data in issue_history.items():
        present = latest_session in data["sessions"]
        base_score = 2 if not present else 0
        tone_score = data.get("tone_score", 0)
        total_score = base_score + tone_score
        results.append({
            "issue": issue,
            "sessions": data["sessions"],
            "duration": data["duration"],
            "intensity": data["intensity"],
            "score": total_score
        })
    return results

def calculate_progress_score(issue_scores):
    total_score = sum([i["score"] for i in issue_scores])
    max_score = len(issue_scores) * 3
    return round((total_score / max_score) * 100, 2) if max_score > 0 else 0

def get_resolution_trend(issue_scores, num_sessions):
    trend = []
    for i in range(num_sessions):
        resolved_count = sum(1 for issue in issue_scores if max(issue["sessions"]) < i)
        trend.append(round((resolved_count / len(issue_scores)) * 100, 2))
    return trend

# ----------- STREAMLIT UI -----------
st.set_page_config(page_title="Enhanced Issue Tracker", layout="wide")
st.title("🧠 Enhanced Patient Issue Tracker with Tone & Intensity")

with st.expander("ℹ️ Instructions"):
    st.markdown("""
    1. Paste one session's JSON guidance at a time.
    2. Tracks issue appearance, tone improvement, and emotional intensity.
    3. Resolution score and trends are shown per session.
    """)

patient_id = st.text_input("🔎 Enter Patient ID")
session_input = st.text_area("📄 Paste Session Guidance (JSON Format)", height=300)

if st.button("📥 Submit Session & Analyze"):
    if not patient_id or not session_input:
        st.warning("Please provide both Patient ID and session JSON.")
    else:
        try:
            session_json = json.loads(session_input)
            new_issues = extract_issues(session_json)
            data = load_data()
            if patient_id not in data:
                data[patient_id] = []
            data[patient_id].append(session_json)
            save_data(data)

            session_issues_list = [extract_issues(s) for s in data[patient_id]]
            issue_scores = score_issues_semantic(session_issues_list)
            progress_score = calculate_progress_score(issue_scores)
            trend = get_resolution_trend(issue_scores, len(session_issues_list))

            st.subheader("📋 Issue Score Table")
            st.dataframe(pd.DataFrame(issue_scores))

            st.metric("📊 Overall Progress Score", f"{progress_score}%")

            st.subheader("📈 Resolution Trend Over Sessions")
            fig, ax = plt.subplots()
            ax.plot([f"S{i+1}" for i in range(len(trend))], trend, marker='o', linestyle='-')
            ax.set_ylim(0, 100)
            ax.set_ylabel("% Resolved")
            ax.set_xlabel("Session")
            ax.set_title("% Issues Resolved Over Time")
            st.pyplot(fig)

            st.success("✅ Session processed and saved!")

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON format. Please check and try again.")
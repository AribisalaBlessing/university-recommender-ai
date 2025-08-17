import streamlit as st
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from datetime import datetime# ========== Load newly extracted data from JAMB brochure==========
df1= pd.read_csv("data_extract.csv", on_bad_lines='skip')

# Clean missing data
df= df1.copy()

# Remove rows with missing course names (critical field)
df= df.dropna(subset=['course'])

# Fill other missing values
df['utme_subjects'] = df['utme_subjects'].fillna('Not available')
df['schools_offering'] = df['schools_offering'].fillna('No schools_offering listed')

# Remove duplicate courses
df= df.drop_duplicates(subset=['course'])

# Reset index after cleaning
df= df.reset_index(drop=True)

# ========== Load Models ==========
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_intent_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

classifier = load_classifier()
intent_model = load_intent_model()

positive_intents = [
    "yes", "sure", "okay", "why not", "go ahead", "i want to see",
    "definitely", "of course", "yeah", "show me"
]

# ========== Streamlit Page Config ==========
st.set_page_config("University Course Bot", "🎓", layout="centered")

# Custom CSS
st.markdown("""
<style>
.user-bubble {
    background-color: #DCF8C6;
    padding: 8px 12px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-end;
}
.bot-bubble {
    background-color: #F1F0F0;
    padding: 8px 12px;
    border-radius: 15px;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-start;
}
.chat-container {
    display: flex;
    flex-direction: column;
}
</style>
""", unsafe_allow_html=True)

# ========== Session State ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "stage" not in st.session_state:
    st.session_state.stage = "start"
if "matched_course" not in st.session_state:
    st.session_state.matched_course = ""
if "log" not in st.session_state:
    st.session_state.log = []

# ========== Functions ==========
def chat(role, msg):
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f"<div class='{bubble_class}'>{msg}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append((role, msg))

def is_positive(text):
    user_vec = intent_model.encode(text, convert_to_tensor=True)
    intents_vec = intent_model.encode(positive_intents, convert_to_tensor=True)
    score = util.cos_sim(user_vec, intents_vec)[0].max().item()
    return score > 0.65

# ========== Show Previous Chat ==========
st.title("🎓 University Course Selection Assistant")
for role, msg in st.session_state.chat_history:
    bubble_class = "user-bubble" if role == "user" else "bot-bubble"
    st.markdown(f"<div class='{bubble_class}'>{msg}</div>", unsafe_allow_html=True)

# ========== Initial Bot Prompt ==========
if st.session_state.stage == "start" and not st.session_state.chat_history:
    chat("assistant", "Hi there! 😊 Tell me what you're interested in studying or your career goals.")

# ========== Input Box ==========
user_input = st.chat_input("Type your message here...")

if user_input:
    chat("user", user_input)

    # Stage 1: Suggest Course
    if st.session_state.stage == "start":
        result = classifier(user_input, df["course"].tolist())
        top_course = result["labels"][0]
        score = result["scores"][0]

        st.session_state.matched_course = top_course
        st.session_state.stage = "confirm_utme"

        chat("assistant", f"It seems like you're interested in *{top_course}*. Would you like to see the UTME requirements?")

        st.session_state.log.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "matched_course": top_course,
            "score": round(score, 4)
        })

    # Stage 2: UTME requirements
    elif st.session_state.stage == "confirm_utme":
        if is_positive(user_input):
            course = st.session_state.matched_course
            utme = df[df["course"] == course]["utme_subjects"].values[0]
            chat("assistant", f"Here are the UTME requirements for *{course}*:<br>{utme}")
            chat("assistant", "Would you also like to see the list of schools_offering offering this course?")
            st.session_state.stage = "confirm_schools_offering"
        else:
            chat("assistant", "Alright! Tell me another interest and I’ll try again. 😊")
            st.session_state.stage = "start"

    # Stage 3: schools_offering
    elif st.session_state.stage == "confirm_schools_offering":
        if is_positive(user_input):
            course = st.session_state.matched_course
            unis = df[df["course"] == course]["schools_offering"].values[0]
            uni_list = "<br>• " + "<br>• ".join(unis.split(","))
            chat("assistant", f"Here are the schools_offering offering *{course}*:<br>{uni_list}")
            chat("assistant", "Was this helpful?")
            st.session_state.stage = "feedback"
        else:
            chat("assistant", "No problem! You can ask me about another course. 😊")
            st.session_state.stage = "start"

    # Stage 4: Feedback
    elif st.session_state.stage == "feedback":
        st.session_state.log[-1]["was_helpful"] = "yes" if is_positive(user_input) else "no"
        chat("assistant", "Thanks for your feedback! 🎓 You can tell me about another interest anytime.")
        st.session_state.stage = "start"

# ========== Download Logs ==========
if st.session_state.log:
    st.markdown("---")
    log_df = pd.DataFrame(st.session_state.log)
    st.download_button("📥 Download Conversation Log", log_df.to_csv(index=False),"logs.csv")

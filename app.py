# app.py — polished Streamlit UI (overwrite your current file)
import streamlit as st
import requests
import json
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"  # update if your backend is running elsewhere
PROPOSAL_PATH = "/mnt/data/Updated project proposal .pdf"  # local path to your uploaded proposal

# ----- Page config -----
st.set_page_config(page_title="Empathetic AI Assistant", page_icon="💬", layout="wide")

# ----- CSS for chat bubbles & badges -----
st.markdown(
    """
<style>
.chat-container { max-width: 900px; margin: 0 auto; }
.user-bubble {
    background: #e6f3ff;
    padding: 10px 14px;
    border-radius: 12px;
    display: inline-block;
    margin: 6px 0;
    max-width: 85%;
    white-space: pre-wrap;
}
.assistant-bubble {
    background: #f7f7fb;
    padding: 10px 14px;
    border-radius: 12px;
    display: inline-block;
    margin: 6px 0;
    max-width: 85%;
    white-space: pre-wrap;
}
.timestamp { font-size: 11px; color: #888; margin-top:4px; }
.badge {
    display:inline-block;
    padding:4px 8px;
    border-radius:999px;
    font-size:12px;
    margin-right:6px;
    margin-top:6px;
}
.badge-joy { background:#FFDE7D; color:#422800; }
.badge-sadness { background:#9EC1FF; color:#05204A; }
.badge-anger { background:#FF9F9F; color:#4A0000; }
.badge-fear { background:#D3B6FF; color:#2D0047; }
.badge-disgust { background:#C8E6C9; color:#10331A; }
.badge-neutral { background:#E0E0E0; color:#222; }
.badge-default { background:#DDD; color:#222; }
.container { padding: 12px; }
.sidebar-note { font-size: 12px; color: #666; }
.header-right { text-align: right; font-size:12px; color:#666; }
</style>
""",
    unsafe_allow_html=True,
)

# ----- Session state -----
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": str, "emotions": [..], "probs": .., "time": ..}

# ----- Helpers -----
def badge_class_for_emotion(emotion: str):
    e = emotion.lower()
    if "joy" in e or "excite" in e: return "badge-joy"
    if "sad" in e or "grief" in e or "remorse" in e: return "badge-sadness"
    if "anger" in e or "annoy" in e: return "badge-anger"
    if "fear" in e or "nervous" in e: return "badge-fear"
    if "disgust" in e: return "badge-disgust"
    if "neutral" in e: return "badge-neutral"
    return "badge-default"

def render_message(msg):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    time = msg.get("time", "")
    emotions = msg.get("emotions", [])
    probs = msg.get("probs", None)

    if role == "user":
        st.markdown(f"<div class='user-bubble'>{content}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='timestamp'>You · {time}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>{content}</div>", unsafe_allow_html=True)
        # emotions badges
        if emotions:
            badges_html = ""
            # If probs provided as full vector, we do not align them by index here (optional improvement)
            for i, emo in enumerate(emotions):
                cls = badge_class_for_emotion(emo)
                score = ""
                # if probs is a list of same length as emotions, show the probability next to the badge
                if isinstance(probs, (list, tuple)) and i < len(probs):
                    try:
                        score = f" {float(probs[i]):.2f}"
                    except Exception:
                        score = ""
                badges_html += f"<span class='badge {cls}'>{emo}{score}</span>"
            st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown(f"<div class='timestamp'>Assistant · {time}</div>", unsafe_allow_html=True)

# ----- Header -----
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("💬 Empathetic AI Assistant")
    st.write("A local emotion-aware assistant — detects emotion and responds empathetically.")
with col2:
    # Link to uploaded project proposal PDF (local path)
    st.markdown(f"<div class='header-right'>Project proposal: <a href='{PROPOSAL_PATH}' target='_blank'>Open PDF</a></div>", unsafe_allow_html=True)
    st.markdown("<div class='header-right'>Tip: use Streamlit settings to switch theme.</div>", unsafe_allow_html=True)

st.write("---")

# ----- Layout: main chat + sidebar -----
main_col, side_col = st.columns([3, 1])

with side_col:
    st.header("Controls")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.experimental_rerun()  # safe here after button
    st.write("")
    st.subheader("Export")
    if st.session_state.messages:
        export_json = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
        st.download_button("Download chat (JSON)", data=export_json, file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
    else:
        st.markdown("<small>No messages to export yet.</small>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Quick tests")
    if st.button("Insert sample (mixed emotions)"):
        sample = "I’m happy for my friend, but deep down I feel a bit jealous and disappointed in myself."
        st.session_state.messages.append({"role":"user","content":sample,"time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        st.experimental_rerun()

with main_col:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display conversation
    for m in st.session_state.messages:
        render_message(m)
        st.write("")

    st.write("---")

    # ---- Input area ----
    # Use a text_area widget that stores value in session_state under 'input_text'
    user_input = st.text_area("Your message", placeholder="How are you feeling today?", key="input_text", height=100)

    # ---- Callback for the Send button ----
    def send_message_callback():
        text = st.session_state.get("input_text", "").strip()
        if not text:
            return

        # Append user message
        st.session_state.messages.append({
            "role": "user",
            "content": text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Call backend and append assistant response
        try:
            with st.spinner("Assistant is typing..."):
                resp = requests.post(API_URL, json={"text": text}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                assistant_text = data.get("llm_response", "(no response)")
                emotions = data.get("emotions", [])
                # backend might provide probabilities under different keys
                probs = data.get("probs") or data.get("probabilities")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_text,
                    "emotions": emotions,
                    "probs": probs,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"[Server Error {resp.status_code}] {resp.text}",
                    "emotions": [],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"API Error: {e}",
                "emotions": [],
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        # Clear the input (allowed inside callback)
        st.session_state.input_text = ""
        # Rerun to display updated chat
        st.rerun()

    # Create Send button that triggers the callback
    st.button("Send", key="send_btn", on_click=send_message_callback)

    st.markdown("</div>", unsafe_allow_html=True)

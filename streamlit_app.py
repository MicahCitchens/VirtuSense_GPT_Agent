import streamlit as st
import openai
from dotenv import load_dotenv
from uuid import uuid4
from datetime import datetime
import os
import time
import re

# === Load env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

# === Client ===
openai.api_key = OPENAI_API_KEY

# === Custom Instructions ===
GPT_ASSISTANT_INSTRUCTIONS = """
You are a technical assistant for diagnosing hardware issues in VirtuSense devices. Use only document chunks retrieved from the vector store to inform your answers. Your job is to return one clearly structured response grounded in real documentation.

Do not generate duplicate sections, multiple formats, or alternate phrasing.

Do not repeat raw document titles (e.g., “Overheating Troubleshoot.pdf”) in your response. Only include document references inside the 📄 Referenced Documents section using metadata.
---

### Response Format

Use this exact format and no other:

## 🛠️ Troubleshooting [Device] — [Symptom Title]

### 🔍 Symptoms of [Symptom]
- List typical signs a user would observe, based on retrieval.

---

## ⚠️ Possible Problems and Solutions  
Use **retrieved document content** to identify real problems and their resolutions.

### 1. **[Problem Title]**
**Problem:** Explain the technical issue based on vector data.  
**Solution:**  
- List exact troubleshooting steps  
- Include escalation triggers if available

### 2. **[Problem Title]**
**Problem:** ...  
**Solution:** ...

---

### 📊 Monitoring and Escalation  
List measurable metrics (e.g., FPS, thermal zones) and clear escalation logic.

---

## 📄 Referenced Documents  
For each referenced chunk, if `doc_url` is present, format the document like this:

📄 [Document Title](https://...)  

Do not use citation numbers, superscripts, or references like [1], [2], or 【#:#†...】. Only show clean markdown links.

If no documents were relevant, say:  
> “I wasn’t able to find supporting documentation for this issue. Would you like me to keep searching or clarify the issue further?”

---

## ✅ Summary  
Summarize the key steps and actions to try. Be concise and direct.

---

### Rules:
- Use only information retrieved from the vector store.
- Do not hallucinate or guess solutions.
- Output a single structured response only.
- Never include citation markers like [1] or 【#:#†...】.
- Prefer retrieved chunks with structured metadata: `extracted_problem`, `extracted_solution`, `title`, `doc_url`.
"""

# === Streamlit Config ===
st.set_page_config(page_title="💬 GPT Support Assistant", layout="centered")
st.title("💬 GPT Support Assistant")

# === Reset button ===
if st.button("🔁 Reset Chat"):
    st.session_state.clear()
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.chat_history = []
    st.session_state.device_name = ""
    st.rerun()

# === Session State Init ===
st.session_state.setdefault("session_id", str(uuid4()))
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("device_name", "")
if "thread_id" not in st.session_state:
    thread = openai.beta.threads.create()
    st.session_state.thread_id = thread.id
if "last_assistant_message_id" not in st.session_state:
    st.session_state.last_assistant_message_id = None

# === Device Selection ===
if not st.session_state.device_name:
    user_device = st.text_input("🔧 Enter your device name (e.g. 'VSTOne', 'VSTBalance', 'VSTAlert+'):")
    if user_device:
        st.session_state.device_name = user_device.strip()
        st.success(f"✅ Device set: {st.session_state.device_name}")
    else:
        st.stop()


# === Display chat history ===
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message, unsafe_allow_html=True)

# === Handle user input ===
query = st.chat_input("Describe the issue or symptom:")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append(("user", query))

    # Submit user message to Assistant thread
    full_user_msg = f"Device: {st.session_state.device_name}\nIssue: {query}"
    openai.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=full_user_msg,
    )

    run = openai.beta.threads.runs.create(
        thread_id=st.session_state.thread_id,
        assistant_id=ASSISTANT_ID,
        instructions=GPT_ASSISTANT_INSTRUCTIONS,
    )

    with st.spinner("Analyzing with internal documentation..."):
        while True:
            status = openai.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id,
            )
            if status.status in ["completed", "failed", "cancelled"]:
                break
            time.sleep(1)

    if status.status == "completed":
        messages = openai.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        for msg in reversed(messages.data):
            if msg.role == "assistant" and msg.id != st.session_state.last_assistant_message_id:
                # Clean auto-citations like   from OpenAI retrieval
                raw_reply = msg.content[0].text.value
                cleaned_reply = re.sub(r"【\d+:\d+†.*?†.*?】", "", raw_reply).strip()

                # Display cleaned response
                with st.chat_message("assistant"):
                    st.markdown(cleaned_reply, unsafe_allow_html=True)

                # Update session state
                st.session_state.chat_history.append(("assistant", cleaned_reply))
                st.session_state.last_assistant_message_id = msg.id
                break
    else:
        st.error("❌ Assistant failed to generate a response.")

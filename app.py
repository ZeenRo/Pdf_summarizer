import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)
import torch
import base64
import os
import requests
import re

# ---- [HARDCODED API KEYS] ----
MISTRAL_API_KEY = "SX19qOYNaafTRsWY0h6YVy2t4DedEqtQ"
GEMINI_API_KEY = "AIzaSyBcP0adNMp_fmixvq9bEB0x0T6J4CFR0Q8"

# ---- Streamlit Config ----
st.set_page_config(layout="wide")
st.title("📄 Document Summarization App (Offline or Online Model)")

# ---- Model Paths and Checkpoints ----
LOCAL_FLAN_DIR = "./local_flan_t5_large"
LOCAL_LAMINI_FLAN_DIR = "./local_lamini_flan_t5_248m"
LOCAL_DISTILBART_DIR = "./local_distilbart_cnn_12_6"
LAMINI_FLAN_CHECKPOINT = "MBZUAI/LaMini-Flan-T5-248M"

# ---- Model Loader Functions ----
@st.cache_resource(show_spinner=True)
def load_flan_t5():
    if not os.path.exists(LOCAL_FLAN_DIR):
        os.makedirs(LOCAL_FLAN_DIR, exist_ok=True)
        checkpoint = "google/flan-t5-large"
        st.info("Downloading google/flan-t5-large model + tokenizer... This will happen only once!")
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        tokenizer.save_pretrained(LOCAL_FLAN_DIR)
        model.save_pretrained(LOCAL_FLAN_DIR)
        st.success(f"Model and tokenizer saved locally at {LOCAL_FLAN_DIR}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(LOCAL_FLAN_DIR)
        model = T5ForConditionalGeneration.from_pretrained(LOCAL_FLAN_DIR)
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_lamini_flan_t5():
    if not os.path.exists(LOCAL_LAMINI_FLAN_DIR):
        os.makedirs(LOCAL_LAMINI_FLAN_DIR, exist_ok=True)
        st.info("Downloading MBZUAI/LaMini-Flan-T5-248M model + tokenizer... This will happen only once!")
        tokenizer = T5Tokenizer.from_pretrained(LAMINI_FLAN_CHECKPOINT)
        model = T5ForConditionalGeneration.from_pretrained(LAMINI_FLAN_CHECKPOINT)
        tokenizer.save_pretrained(LOCAL_LAMINI_FLAN_DIR)
        model.save_pretrained(LOCAL_LAMINI_FLAN_DIR)
        st.success(f"Model and tokenizer saved locally at {LOCAL_LAMINI_FLAN_DIR}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(LOCAL_LAMINI_FLAN_DIR)
        model = T5ForConditionalGeneration.from_pretrained(LOCAL_LAMINI_FLAN_DIR)
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_bart():
    checkpoint = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model
























# ==== BEGIN PEGASUS XSUM LOADER ====
@st.cache_resource(show_spinner=True)
def load_pegasus():
    checkpoint = "google/pegasus-xsum"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model
# ==== END PEGASUS XSUM LOADER ====







# ==== BEGIN DISTILBART-CNN-12-6 LOADER ====
@st.cache_resource(show_spinner=True)
def load_distilbart():
    """Load or download DistilBART-CNN-12-6 model and tokenizer."""
    if not os.path.exists(LOCAL_DISTILBART_DIR):
        os.makedirs(LOCAL_DISTILBART_DIR, exist_ok=True)
        checkpoint = "sshleifer/distilbart-cnn-12-6"
        st.info("Downloading distilbart-cnn-12-6 model + tokenizer... This will happen only once!")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        tokenizer.save_pretrained(LOCAL_DISTILBART_DIR)
        model.save_pretrained(LOCAL_DISTILBART_DIR)
        st.success(f"Model and tokenizer saved locally at {LOCAL_DISTILBART_DIR}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_DISTILBART_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_DISTILBART_DIR)
    return tokenizer, model
# ==== END DISTILBART-CNN-12-6 LOADER ====











# ==== BEGIN PEGASUS XSUM SUMMARIZATION ====
def pegasus_summarize(filepath):
    tokenizer, model = load_pegasus()
    input_chunks = file_preprocessing(filepath)
    all_summaries = []
    for chunk in input_chunks:
        # Prompt for paragraph + points
        prompt = (
            "Summarize the following text with a concise paragraph. "
            "If appropriate, add 3-5 key points as bullet points after the paragraph.\n\n"
            f"{chunk}"
        )
        inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=2,              # Faster decoding
                max_length=80,            # Slightly longer for paragraph+points, but still fast
                min_length=25,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        all_summaries.append(summary)
    return "\n\n".join(all_summaries)
# ==== END PEGASUS XSUM SUMMARIZATION ====





















# ==== BEGIN DISTILBART-CNN-12-6 SUMMARIZATION ====
def distilbart_summarize(filepath):
    tokenizer, model = load_distilbart()
    input_chunks = file_preprocessing(filepath)
    all_summaries = []
    pipe_sum = pipeline(
        'summarization',
        model=model,
        tokenizer=tokenizer,
        max_length=90,    # Shorter, more focused
        min_length=30
    )
    for chunk in input_chunks:
        if not chunk.strip():
            continue
        # Prompt for a professional summary with points
        prompt = (
            "Summarize the following text in a professional manner. "
            "Begin with a concise paragraph, then—if appropriate—add 3-5 key points as bullet points.\n\n"
            f"{chunk}"
        )
        try:
            result = pipe_sum(prompt)
            summary = result[0]['summary_text'].strip()
            if not summary:
                summary = "[No summary generated for this chunk.]"
            all_summaries.append(summary)
        except Exception as e:
            all_summaries.append(f"[Error summarizing chunk: {e}]")
    return "\n\n".join(all_summaries)



# ==== END DISTILBART-CNN-12-6 SUMMARIZATION ====




























# ---- File Preprocessing ----
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    full_text = ''.join([page.page_content for page in pages])
    total_length = len(full_text)
    target_chunks = 10
    chunk_size = max(300, total_length // target_chunks)
    chunk_overlap = chunk_size // 5
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(pages)
    chunks = [text.page_content for text in texts]
    return chunks

# ---- Summarization ----
def get_length_params(length_choice, model_type='flan'):
    if length_choice == "Small":
        min_ratio, max_ratio = 0.1, 0.2
        min_cap, max_cap = 20, 50
    elif length_choice == "Medium":
        min_ratio, max_ratio = 0.2, 0.5
        min_cap, max_cap = 40, 100
    else:
        min_ratio, max_ratio = 0.4, 0.8
        min_cap, max_cap = 80, 200
    if model_type == 'bart':
        max_cap = min(max_cap, 512)
    return min_ratio, max_ratio, min_cap, max_cap

def offline_summarize_with_model(filepath, length_choice, tokenizer, model):
    input_chunks = file_preprocessing(filepath)
    all_summaries = []
    min_ratio, max_ratio, min_cap, max_cap = get_length_params(length_choice, 'flan')
    for chunk in input_chunks:
        input_len = len(tokenizer(chunk, truncation=True)['input_ids'])
        max_len = min(max_cap, int(input_len * max_ratio))
        min_len = max(min_cap, int(input_len * min_ratio))
        pipe_sum = pipeline(
            'summarization',
            model=model,
            tokenizer=tokenizer,
            max_length=max_len,
            min_length=min_len
        )
        result = pipe_sum(chunk, truncation=True)
        all_summaries.append(result[0]['summary_text'])
    return "\n\n".join(all_summaries)

def online_summarize(filepath, length_choice, online_model="BART"):
    input_chunks = file_preprocessing(filepath)
    all_summaries = []




































    # ---- [PARAGRAPH + BULLET POINTS] Mistral API Summarization ----
    def summarize_with_mistral(text, summary_length):
        if not MISTRAL_API_KEY:
            return "[Mistral API ERROR]: API key not set."
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        word_limits = {"Small": 100, "Medium": 200, "Large": 350}
        word_limit = word_limits.get(summary_length, 200)
        max_input_words = 3000
        input_words = text.split()
        if len(input_words) > max_input_words:
            text = " ".join(input_words[:max_input_words])
        prompt = (
            f"Write a precise summary of the following text in clear, fluent English, "
            f"using no more than {word_limit} words. Begin with a short summary paragraph, "
            f"then list the 4-7 most important points as bullet points starting with a dash (-). "
            f"Bullet points should be concise and highlight key facts, objectives, or outcomes.\n\n"
            f"Text:\n{text}"
        )
        payload = {
            "model": "mistral-large-latest",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": word_limit * 2,
            "temperature": 0.5
        }
        response = None
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            summary = data['choices'][0]['message']['content']
            # Post-process: ensure first paragraph, then points
            lines = [line.strip() for line in summary.split('\n') if line.strip()]
            para = []
            bullets = []
            in_bullets = False
            for line in lines:
                if line.startswith('-'):
                    in_bullets = True
                    bullets.append(line)
                elif in_bullets:
                    # If bullet points are interrupted by a non-bullet, treat as extra bullet
                    bullets.append('- ' + line.lstrip('-•* '))
                else:
                    para.append(line)
            # Trim to word limit
            all_text = " ".join(para + bullets)
            words = all_text.split()
            if len(words) > word_limit:
                # Try to keep paragraph and as many bullets as possible
                trimmed_para = []
                trimmed_bullets = []
                count = 0
                for line in para:
                    line_words = line.split()
                    if count + len(line_words) > word_limit:
                        break
                    trimmed_para.append(line)
                    count += len(line_words)
                for line in bullets:
                    line_words = line.split()
                    if count + len(line_words) > word_limit:
                        break
                    trimmed_bullets.append(line)
                    count += len(line_words)
                trimmed_bullets.append("...")
                return "\n".join(trimmed_para + [""] + trimmed_bullets)
            return "\n".join(para + [""] + bullets)
        except Exception as e:
            error_msg = f"[Mistral API ERROR]: {e}"
            if response is not None:
                error_msg += f"\nStatus: {response.status_code}\nResponse: {response.text}"
            return error_msg

    # ---- [PARAGRAPH + BULLET POINTS] Gemini API Summarization ----
    def summarize_with_gemini(text, summary_length):
        if not GEMINI_API_KEY:
            return "[Gemini API ERROR]: API key not set."
        url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        word_limits = {"Small": 100, "Medium": 200, "Large": 350}
        word_limit = word_limits.get(summary_length, 200)
        max_input_words = 3000
        input_words = text.split()
        if len(input_words) > max_input_words:
            text = " ".join(input_words[:max_input_words])
        prompt = (
            f"Write a precise summary of the following text in clear, fluent English, "
            f"using no more than {word_limit} words. Begin with a short summary paragraph, "
            f"then list the 4-7 most important points as bullet points starting with a dash (-). "
            f"Bullet points should be concise and highlight key facts, objectives, or outcomes.\n\n"
            f"Text:\n{text}"
        )
        payload = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }
        response = None
        try:
            response = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            summary = data["candidates"][0]["content"]["parts"][0]["text"]
            # Post-process: ensure first paragraph, then points
            lines = [line.strip() for line in summary.split('\n') if line.strip()]
            para = []
            bullets = []
            in_bullets = False
            for line in lines:
                if line.startswith('-'):
                    in_bullets = True
                    bullets.append(line)
                elif in_bullets:
                    bullets.append('- ' + line.lstrip('-•* '))
                else:
                    para.append(line)
            # Trim to word limit
            all_text = " ".join(para + bullets)
            words = all_text.split()
            if len(words) > word_limit:
                trimmed_para = []
                trimmed_bullets = []
                count = 0
                for line in para:
                    line_words = line.split()
                    if count + len(line_words) > word_limit:
                        break
                    trimmed_para.append(line)
                    count += len(line_words)
                for line in bullets:
                    line_words = line.split()
                    if count + len(line_words) > word_limit:
                        break
                    trimmed_bullets.append(line)
                    count += len(line_words)
                trimmed_bullets.append("...")
                return "\n".join(trimmed_para + [""] + trimmed_bullets)
            return "\n".join(para + [""] + bullets)
        except Exception as e:
            error_msg = f"[Gemini API ERROR]: {e}"
            if response is not None:
                error_msg += f"\nStatus: {response.status_code}\nResponse: {response.text}"
            return error_msg

    # ---- [ORIGINAL] BART Summarization ----
    if online_model == "BART":
        tokenizer, model = load_bart()
        min_ratio, max_ratio, min_cap, max_cap = get_length_params(length_choice, 'bart')
        for chunk in input_chunks:
            input_len = len(tokenizer(chunk, truncation=True)['input_ids'])
            max_len = min(max_cap, int(input_len * max_ratio), 512)
            min_len = max(min_cap, int(input_len * min_ratio))
            pipe_sum = pipeline(
                'summarization',
                model=model,
                tokenizer=tokenizer,
                max_length=max_len,
                min_length=min_len
            )
            result = pipe_sum(chunk, truncation=True)
            all_summaries.append(result[0]['summary_text'])
    # ---- [WORD-LIMITED] Mistral Summarization ----
    elif online_model == "Mistral":
        summary = summarize_with_mistral(" ".join(input_chunks), length_choice)
        all_summaries.append(summary)
    # ---- [WORD-LIMITED] Gemini Summarization ----
    elif online_model == "Gemini":
        summary = summarize_with_gemini(" ".join(input_chunks), length_choice)
        all_summaries.append(summary)
    else:
        all_summaries.append("Unknown online model selected.")
    return "\n\n".join(all_summaries)




































# ---- PDF Display ----
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os

def displayPDF(file_path, width=600, height=400):
    with open(file_path, "rb") as f:
        binary_data = f.read()

    st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            max-width: {width + 20}px;
            margin: auto;
        ">
            <h4 style="text-align:center; color:#333;">📄 Document Preview</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pdf_viewer(input=binary_data, width=width, height=height)






# ---- UI ----
mode_choice = st.radio(
    "Select summarization mode:",
    ("Offline", "Online"),
    index=0,
    help="Offline runs locally after first download. Online uses BART (requires internet unless cached)."
)

offline_model = None
online_model = None  # [ADDED]
if mode_choice == "Offline":
    offline_model = st.selectbox(
        "Select offline model:",
        ("Flan-T5-Large", "LaMini-Flan-T5-248M", "Pegasus XSUM", "DistilBART-CNN-12-6"),  # <--- Added here
        index=0
    )
else:
    # ---- [ADDED] Online model selection ----
    online_model = st.selectbox(
        "Select online model:",
        ("BART", "Mistral", "Gemini"),
        index=0
    )

summary_length = st.radio(
    "Select summary length:",
    ("Small", "Medium", "Large"),
    index=1,
    help="Small: Shortest summary, Medium: Balanced, Large: Most detailed"
)

uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

if uploaded_file is not None:
    if st.button("Summarize"):
        col1, col2 = st.columns(2)
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        with col1:
            st.info("Uploaded File")
            displayPDF(filepath)

        with col2:
            with st.spinner("Generating summary..."):
                if mode_choice == "Offline":
                    if offline_model == "Flan-T5-Large":
                        tokenizer, model = load_flan_t5()
                        summary = offline_summarize_with_model(filepath, summary_length, tokenizer, model)
                    elif offline_model == "LaMini-Flan-T5-248M":
                        tokenizer, model = load_lamini_flan_t5()
                        summary = offline_summarize_with_model(filepath, summary_length, tokenizer, model)
                    elif offline_model == "Pegasus XSUM":
                        summary = pegasus_summarize(filepath)
                    elif offline_model == "DistilBART-CNN-12-6":
                        summary = distilbart_summarize(filepath)
                else:
                    # ---- [ADDED] Pass selected online model ----
                    summary = online_summarize(filepath, summary_length, online_model)
            st.success("✅ Summarization Complete")
            st.markdown(summary)
            st.download_button(
                label="📥 Download Summary as .txt",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

import streamlit as st
import fitz  # PyMuPDF for PDF processing
from groq import Groq
import os
import pymupdf
import re
from openai import OpenAI
import textwrap
import time
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh
from supabase import create_client, Client
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import platform
from datetime import datetime, timezone
import logging
import sys
import html
import re as _re


# ---------- Logging (keep helpful, non-intrusive) ---------------------------------
def _init_debug_logging():
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stderr)],
        )
    except Exception:
        pass
    try:
        st.set_option("logger.level", "debug")
        st.set_option("client.showErrorDetails", False)
    except Exception:
        pass


def _log_exc(msg, e=None):
    try:
        if e is not None:
            logging.exception(f"{msg}: {e}")
        else:
            logging.exception(msg)
    except Exception:
        pass


_init_debug_logging()


# ---------- Groq client with ASCII-only headers (avoid httpx header errors) ------
GROQ_KEYS_URL = (
    "https://console.groq.com/keys?_gl=1*129xulo*_gcl_au*MTMyNzU2Njk3Ny4xNzU5MzA0MzU5*_ga*NTk2NDgyMDAzLjE3NTkzMDQzNTk."
    "*_ga_4TD0X2GEZG*czE3NjIzMjYzNTYkbzQkZzEkdDE3NjIzMjYzOTYkajIwJGwwJGgw"
)

if "user_groq_api_key" not in st.session_state:
    st.session_state["user_groq_api_key"] = ""


def _normalize_api_key(raw: str) -> str:
    try:
        return (raw or "").strip()
    except Exception:
        return ""


def _remember_user_api_key(raw: str) -> None:
    st.session_state["user_groq_api_key"] = _normalize_api_key(raw)


def _resolve_groq_api_key() -> str:
    key = _normalize_api_key(st.session_state.get("user_groq_api_key", ""))
    if key.startswith("gsk_"):
        return key
    try:
        fallback = st.secrets.get("groq", {}).get("api_key", "")
    except Exception:
        fallback = ""
    return _normalize_api_key(fallback)


def _sanitize_ascii(value: str) -> str:
    try:
        return value.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return ""


class SafeGroq(Groq):
    @property
    def default_headers(self) -> dict:
        base = super().default_headers
        safe = {k: _sanitize_ascii(str(v)) for k, v in base.items()}
        safe.update(
            {
                "User-Agent": "groq-python",
                "X-Stainless-OS": "Windows",
                "X-Stainless-Arch": "x64",
                "X-Stainless-Runtime": "CPython",
                "X-Stainless-Runtime-Version": _sanitize_ascii(platform.python_version() or "3"),
            }
        )
        return safe


def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)

# ============================== Auth (Streamlit built-in) ========================
if not st.user.is_logged_in:
    st.markdown(
        """
        <style>
        .header-card { text-align: center; padding: 20px; margin-bottom: 15px; border-radius: 12px;
            background: linear-gradient(to right, #1E1A4D, #440E03); box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            border: 3px solid #1C398E; }
        .header-card h3 { margin: 0; font-weight: 700; color: #E2E8F0; }
        .header-card h6 { margin-top: 6px; font-weight: 500; color: #E2E8F0; }
        </style>
        <div class="header-card">
            <h3>Welcome to the Academic Student Tutorial and Excellence Programme (A_STEP)</h3>
            <h6>A UFS student driven academic support and development initiative</h6>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Slideshow
    image_urls = [
        "https://i.postimg.cc/BQsN9j4F/students3.jpg",
        "https://i.postimg.cc/4xY9rG7H/students2.jpg",
        "https://i.postimg.cc/bJgrZVSk/students1.jpg",
    ]
    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0
    st_autorefresh(interval=5000, key="slideshow_refresh")
    st.empty().image(image_urls[st.session_state.slide_index], use_container_width=True)
    st.session_state.slide_index = (st.session_state.slide_index + 1) % len(image_urls)

    st.markdown("### Step 1: Bring your own Groq API key")
    st.markdown(
        f"""
        - Get a free key from [Groq Console]({GROQ_KEYS_URL}) (opens in a new tab).
        - Paste it below — the app keeps it only in this browser session.
        """,
        unsafe_allow_html=False,
    )
    api_key_before_login = st.text_input(
        "Paste your Groq API key",
        value=st.session_state.get("user_groq_api_key", ""),
        placeholder="gsk_...",
        type="password",
        key="groq_api_key_before_login",
        help="Create a key on Groq, then paste it here to use your own quota.",
    )
    _remember_user_api_key(api_key_before_login)

    saved_key = st.session_state.get("user_groq_api_key", "")
    has_valid_key = saved_key.startswith("gsk_")  # Groq API keys currently start with gsk_
    if saved_key:
        if has_valid_key:
            st.success("Groq API key saved — you can sign in now.")
    else:
        st.info("Need help? Click the link above to open the Groq console and copy your key.")

    if has_valid_key:
        st.markdown("### Step 2: Sign in with your UFS account")
        st.markdown(
            """
            <div style="background-color: #1a1a1a; padding: 10px; border-radius: 8px; border: 2px solid white;">
                <p style="color: white; font-size: 16px; text-align: center;">
                    Please sign in using your UFS Google Student Account (ufs4life) to access the GenAI Assistant Tutor.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            if st.button("Sign-in"):
                #st.login("google")
                login_screen()

else:
    # ============================== Logged-in ====================================
    _name = getattr(st.user, "name", None)
    _email = getattr(st.user, "email", None)
    if _name and _email:
        st.sidebar.success(f"Welcome, {_name} {_email}!")
    elif _email:
        st.sidebar.success(f"Welcome, {_email}!")
    else:
        st.sidebar.success("Welcome!")

    # -------- Access logging to Supabase (login + logout) ------------------------
    def _report_access_log_status(message: str, level: str = "info"):
        try:
            if level in ("error", "warning"):
                return
            if level == "success":
                st.sidebar.success(message)
            else:
                st.sidebar.info(message)
        except Exception as e:
            _log_exc("_report_access_log_status failed", e)

    def _get_root_supabase_keys():
        try:
            url = str(st.secrets.get("SUPABASE_URL", "")).strip()
            key = ""
            for kname in ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY"):
                if not key:
                    key = str(st.secrets.get(kname, "")).strip()
            if url and key:
                return url, key
        except Exception as e:
            _log_exc("top-level SUPABASE secrets read failed", e)

        try:
            sb = st.secrets.get("supabase", {}) or {}
            cand_urls = [sb.get("url"), sb.get("URL"), sb.get("supabase_url"), sb.get("SUPABASE_URL")]
            cand_keys = [
                sb.get("service_role_key"), sb.get("SERVICE_ROLE_KEY"),
                sb.get("supabase_service_role_key"), sb.get("SUPABASE_SERVICE_ROLE_KEY"),
                sb.get("anon_key"), sb.get("ANON_KEY"),
            ]
            url = (next((x for x in cand_urls if x), "") or "").strip()
            key = (next((x for x in cand_keys if x), "") or "").strip()
            if url and key:
                return url, key
        except Exception as e:
            _log_exc("[supabase] table fallback failed", e)

        url = (os.environ.get("SUPABASE_URL") or "").strip()
        key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or "").strip()
        if url and key:
            return url, key

        try:
            vec_cfg = st.secrets.get("vectors", {}) or {}
            url = (vec_cfg.get("SUPABASE_URL1") or vec_cfg.get("url") or "").strip()
            key = (vec_cfg.get("SUPABASE_KEY1") or vec_cfg.get("key") or "").strip()
            if url and key:
                return url, key
        except Exception:
            pass
        return "", ""

    def _get_access_supabase_client():
        try:
            url, key = _get_root_supabase_keys()
            if not url or not key:
                return None
            return create_client(url, key)
        except Exception as e:
            _log_exc("Failed to init Supabase client", e)
            return None

    def _record_login_if_needed(supabase_client, email, name):
        if not supabase_client or not email or st.session_state.get("access_log_id"):
            return
        ts = datetime.now(timezone.utc).isoformat()
        meta = {"event": "login", "login_ts": ts, "app": "A_STEP_Tutor"}
        try:
            resp = supabase_client.table("user_access_logs").insert(
                {"user_email": email, "user_name": name or None, "meta": meta}
            ).execute()
            data = getattr(resp, "data", None)
            if data and isinstance(data, list) and data:
                st.session_state["access_log_id"] = data[0].get("id")
                st.session_state["access_meta"] = meta
                st.session_state["session_started_at"] = time.time()
                # success toast intentionally suppressed
        except Exception as e:
            _log_exc("Access log insert failed", e)

    def _record_logout_update(supabase_client):
        if not supabase_client:
            return
        log_id = st.session_state.get("access_log_id")
        started = st.session_state.get("session_started_at")
        if not log_id:
            return
        try:
            duration = int(time.time() - started) if started else None
            ts = datetime.now(timezone.utc).isoformat()
            meta = dict(st.session_state.get("access_meta", {}))
            meta.update({"logout_ts": ts, "duration_sec": duration, "event": "logout"})
            supabase_client.table("user_access_logs").update({"meta": meta}).eq("id", log_id).execute()
            _report_access_log_status("Access log updated with logout time.", level="success")
        except Exception as e:
            _log_exc("Access log update failed", e)

    _access_sb = _get_access_supabase_client()
    _record_login_if_needed(_access_sb, _email, _name)

    if st.sidebar.button("Lesson Over? Sign-Out Here!", key="signout_button"):
        _record_logout_update(_access_sb)
        st.logout()
        st.session_state.clear()
        st.rerun()

    # ============================== GenAI API (Groq) ============================
    api_key = _resolve_groq_api_key()
    if not api_key or not api_key.startswith("gsk_"):
        st.stop()

    py_ver = _sanitize_ascii(platform.python_version())
    arch = _sanitize_ascii("x64" if "64" in (platform.machine() or "") else "x32")
    os_name = _sanitize_ascii("Windows" if platform.system().lower() == "windows" else platform.system())
    default_headers = {
        "User-Agent": "groq-python",
        "X-Stainless-OS": os_name or "Windows",
        "X-Stainless-Arch": arch or "x64",
        "X-Stainless-Runtime": "CPython",
        "X-Stainless-Runtime-Version": py_ver or "3",
    }
    client = SafeGroq(api_key=api_key, default_headers=default_headers)

    def _get_vectors_supabase_client():
        try:
            vec_cfg = st.secrets.get("vectors", {}) or {}
            use_separate = bool(vec_cfg.get("use_separate_project", False))
            url1 = (vec_cfg.get("SUPABASE_URL1") or "").strip()
            key1 = (vec_cfg.get("SUPABASE_KEY1") or "").strip()
            if use_separate and url1 and key1:
                return create_client(url1, key1)
            url, key = _get_root_supabase_keys()
            return create_client(url, key) if url and key else None
        except Exception as e:
            # _log_exc("Failed to init vectors Supabase client", e)
            return None

    supabase: Client = _get_vectors_supabase_client()

    # ============================== UI Chrome ===================================
    st.markdown(
        """
        <style>
        .header-card { text-align: center; padding: 20px; margin-bottom: 15px; border-radius: 12px;
            background: linear-gradient(to right, #1E1A4D, #440E03); box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            border: 3px solid #1C398E; }
        .header-card h3 { margin: 0; font-weight: 700; color: #E2E8F0; }
        .header-card h6 { margin-top: 6px; font-weight: 500; color: #E2E8F0; }
        </style>
        <div class="header-card">
          <h3>A_STEP GenAI Assistant Tutor</h3>
          <h6>For real-time, all access, personalised and adaptive learning</h6>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("![Alt Text](https://i.postimg.cc/dtqz6njz/log.png)")

    # Hide any previously injected main-page nav components (avatar/cards) if present
    st.markdown(
        """
        <style>
        .nav-wrap, .card-btn, .outline-btn { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation (revert to original controls)
    st.sidebar.markdown("---")
    genre = st.sidebar.radio(
        "Select your Preferred Learning Mode",
        [":rainbow[Tutor Session Mode]", "***Material Engagement***"],
        captions=["Engagement with a GenAI tutor", "Material Assistance with GenAI"],
    )
    new_chat = st.sidebar.button("Clear or Start a New Chat!", key="clear_new_chat_sidebar")

    
    # Convert LaTeX delimiters and render with markdown so math shows nicely
    LATEX_CODE_BLOCK_RE = _re.compile(r"```.*?```", _re.DOTALL)
    def _convert_tex_delimiters(text: str) -> str:
        try:
            parts = LATEX_CODE_BLOCK_RE.split(text)
            fences = LATEX_CODE_BLOCK_RE.findall(text)
            out = []
            for i, seg in enumerate(parts):
                s = _re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", seg, flags=_re.DOTALL)
                s = _re.sub(r"\\\((.*?)\\\)", r"$\1$", s)
                # Normalize double backslashes often used for line breaks
                s = s.replace("\\n", "\n")
                out.append(s)
                if i < len(fences):
                    out.append(fences[i])
            return "".join(out)
        except Exception:
            return text

    def _strip_html_tags_keep_structure(text: str) -> str:
        try:
            t = text
            t = _re.sub(r"<br\s*/?>", "\n", t, flags=_re.IGNORECASE)
            t = _re.sub(r"</p>\s*", "\n\n", t, flags=_re.IGNORECASE)
            t = _re.sub(r"<p[^>]*>", "", t, flags=_re.IGNORECASE)
            for i in range(6, 0, -1):
                t = _re.sub(rf"<h{i}[^>]*>(.*?)</h{i}>", lambda m: "#"*i + " " + m.group(1) + "\n\n", t, flags=_re.IGNORECASE|_re.DOTALL)
            t = _re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", t, flags=_re.IGNORECASE|_re.DOTALL)
            t = _re.sub(r"<(em|i)>(.*?)</\1>", r"*\2*", t, flags=_re.IGNORECASE|_re.DOTALL)
            t = _re.sub(r"<pre><code>(.*?)</code></pre>", lambda m: "```\n" + m.group(1) + "\n```", t, flags=_re.IGNORECASE|_re.DOTALL)
            t = _re.sub(r"<code>(.*?)</code>", r"`\1`", t, flags=_re.IGNORECASE|_re.DOTALL)
            t = _re.sub(r"\s*</li>\s*", "\n", t, flags=_re.IGNORECASE)
            t = _re.sub(r"<li[^>]*>", "- ", t, flags=_re.IGNORECASE)
            t = _re.sub(r"</?(ul|ol)[^>]*>", "\n", t, flags=_re.IGNORECASE)
            def table_to_md(m):
                tb = m.group(1)
                rows = _re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=_re.IGNORECASE|_re.DOTALL)
                md_rows = []
                headers = []
                for idx, row in enumerate(rows):
                    th = _re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=_re.IGNORECASE|_re.DOTALL)
                    cells = [ _re.sub(r"<[^>]+>", "", c).strip() for c in th ]
                    if idx == 0:
                        headers = cells if cells else []
                    md_rows.append("| " + " | ".join(cells) + " |")
                if headers:
                    sep = "| " + " | ".join(["---"]*len(headers)) + " |"
                    return md_rows[0] + "\n" + sep + ("\n" + "\n".join(md_rows[1:]) if len(md_rows)>1 else "")
                else:
                    return "\n".join(md_rows)
            t = _re.sub(r"<table[^>]*>(.*?)</table>", table_to_md, t, flags=_re.IGNORECASE|_re.DOTALL)
            t = _re.sub(r"<[^>]+>", "", t)
            return t
        except Exception:
            return text

    def _normalize_model_output(text: str) -> str:
        parts = LATEX_CODE_BLOCK_RE.split(text)
        fences = LATEX_CODE_BLOCK_RE.findall(text)
        out = []
        for i, seg in enumerate(parts):
            seg = _strip_html_tags_keep_structure(seg)
            seg = _convert_tex_delimiters(seg)
            out.append(seg)
            if i < len(fences):
                out.append(fences[i])
        return "".join(out)

    def render_bubble(role: str, content: str):
        txt = _normalize_model_output(content or "")
        with st.chat_message(role):
            st.markdown(txt)



    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <style>
                .stButton>button { background-color: maroon; color: white; font-size: 16px; padding: 10px 24px;
                    border: none; border-radius: 8px; cursor: pointer; }
                .stButton>button:hover { background-color: #45a049; }
            </style>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================== Material Engagement =========================
    if genre == "***Material Engagement***":
        template = """
            Act as the **A_STEP GenAI Assistant Tutor** for the *Academic Student Excellence and Tutorial Programme (A_STEP)* 
            at the **University of the Free State (UFS)** in South Africa.  
            You specialise in helping first-year students understand and engage with **academic learning materials** — 
            such as PDFs, notes, study guides, and readings — to deepen their comprehension and learning skills.

            You are **supportive, professional, and friendly**, like a real tutor with strong subject knowledge, empathy, and a teaching mindset.  
            Your role is to *guide*, *explain*, and *coach* — **not** to directly solve or provide answers to assignments, tests, or take-home problems.

            ---

            ### 💡 Core Teaching Philosophy

            1. **Purpose**
               - Help students explore, interpret, and understand the material they upload.
               - Never provide direct answers to questions that resemble assignments, essays, or test prompts.
               - Instead, guide the student to reach understanding on their own through conceptual hints, examples, and Socratic questioning.
               - Encourage reflection, analysis, and application of ideas from the uploaded material.

            2. **Interaction Style**
               - Greet warmly, introduce yourself briefly at the start of a new session (once only).
               - Communicate in **clear, plain English** appropriate for first-year university students.
               - Avoid academic jargon unless explaining its meaning.
               - Guide learning in **small, progressive steps** and check for understanding after key points.
               - Encourage active participation — e.g., “What do you think this section is trying to say?” or “Can you identify the main idea in this paragraph?”

            3. **Ethical Tutoring Approach**
               - Never give full solutions to assignment questions, tests, or problem sets.
               - Instead:
                   - Rephrase the question to help the student think critically.
                   - Offer examples or frameworks they can apply on their own.
                   - Explain related theories or principles in general terms.
                   - Encourage them to attempt a response and provide constructive feedback.
               - If a user insists on an answer, politely remind them that your role is to *guide learning*, not to provide completed academic work.

            4. **Working with Uploaded Materials (PDFs)**
               - Use `{pdf_content}` as reference material for context.
               - Avoid engaging with topics or context outside the pdf_content material. Steer the student back to the uploaded context.
               - Summarize or explain concepts found in the uploaded content.
               - Help the student identify key themes, definitions, or examples.
               - Provide structure (e.g., outlines, key takeaways, or concept maps) that aids understanding.
               - When relevant, connect the material to broader academic principles or South African educational contexts.

            5. **Output Structure**
               - **Explanations:** Use concise bullet points or short paragraphs.
               - **Concept Mapping:** Break complex topics into main ideas and subtopics.
               - **IRAC or Framework Analysis:** For law, business, or applied topics, use *Issue – Rule – Application – Conclusion*.
               - **Study Skills Support:** Offer guidance like “How to summarize this section” or “Tips for remembering these definitions.”
               - Avoid LaTeX formatting. Use markdown (e.g., bold, bullet lists, emojis).

            6. **Engagement & Quizzes**
               - When appropriate, generate a short 5-question quiz (multiple-choice or short-answer) **about the uploaded material**.
               - The goal is reinforcement — not assessment.
               - If the student answers incorrectly, provide gentle feedback and an example before asking them to retry.
               - Praise progress to keep motivation high.

            7. **Tone & Personality**
               - Friendly, encouraging, and empathetic.
               - Use emojis sparingly but effectively (e.g., 🌱📘✨).
               - Foster curiosity, not dependency.

            8. **Response Structure**
               - Begin with a brief acknowledgment of the topic or question.
               - Reference relevant concepts from the uploaded PDF.
               - Offer guidance and explanation.
               - End with a *Next Steps* suggestion (e.g., “Try summarizing this paragraph in your own words — I can help you check it 👇”).

            ---

            **PDF Content:**
            {pdf_content}

            **Conversation History:**
            {context}

            **Student’s Question:**
            {question}

            **Answer:**
            """

        def extract_text_from_pdf(pdf_file):
            pdf_reader = pymupdf.open(stream=pdf_file.getvalue(), filetype="pdf")
            text = ""
            for page in pdf_reader:
                text += page.get_text("text") + "\n"
            return text

        st.sidebar.markdown("<h1 style='text-align: center;'>Upload PDFs</h1>", unsafe_allow_html=True)
        uploaded_file = st.sidebar.file_uploader(" ", type=["pdf"])

        if new_chat:
            st.session_state.messages = []
            st.session_state.pdf_content = ""
            st.success("New chat started! Upload a new PDF if needed.")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_content" not in st.session_state:
            st.session_state.pdf_content = ""

        if uploaded_file is not None:
            st.session_state.pdf_content = extract_text_from_pdf(uploaded_file)
            st.sidebar.success("PDF uploaded successfully!")

        for message in st.session_state.messages:
            render_bubble(message["role"], message["content"])

        user_input = st.chat_input("Ask something...")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            render_bubble("user", user_input)

            context_messages = st.session_state.messages[-5:]
            context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in context_messages])

            if not st.session_state.pdf_content:
                response = (
                    "Hi there 👋. Welcome to the ***Material Engagement*** Tutorial Session. I'm your A_STEP Assistant tutor ✨. "
                    "I see that no PDF document has been uploaded yet 🤷. "
                    "Use the upload button to choose a PDF 📖, then we can proceed with your questions about it — or switch to the ***Tutor Session Mode*** to chat with a GenAI Tutor 🧑‍🏫."
                )
            else:
                MAX_PDF_CHARS = 6000
                pdf_text = st.session_state.pdf_content
                if len(pdf_text) > MAX_PDF_CHARS:
                    pdf_text = pdf_text[:MAX_PDF_CHARS] + "\n[TRUNCATED]"
                prompt_text = template.format(pdf_content=pdf_text, context=context, question=user_input)
                safe_headers = {
                    "User-Agent": "groq-python",
                    "X-Stainless-OS": "Windows",
                    "X-Stainless-Arch": "x64",
                    "X-Stainless-Runtime": "CPython",
                    "X-Stainless-Runtime-Version": "3",
                }
                try:
                    groq_response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=0.7,
                        max_tokens=2000,
                        extra_headers=safe_headers,
                    )
                    raw_text = groq_response.choices[0].message.content
                    response = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                except Exception:
                    st.stop()

        st.session_state.messages.append({"role": "assistant", "content": response})
        render_bubble("assistant", response)

    # ============================== Tutor Session Mode ==========================
    else:
        template = """
            You are the **A_STEP GenAI Assistant Tutor**, part of the *Academic Student Excellence and Tutorial Programme (A_STEP)* at the University of the Free State (UFS), South Africa.
            You specialise in supporting **first-year university students** across all faculties. 
            You are friendly, empathetic, and professional — like a real tutor who understands the challenges of transitioning to university life.

            Your purpose is to help students understand and engage with academic material through guided discussion, critical questioning, and interactive learning activities.

            **Important:** Use the relevant course/module information provided (from the embedded course data) to support your answers whenever possible.


            Follow the structure below when responding:

            1. **Context & Scope**  
               - Focus on the South African university context (UFS).  
               - Cover topics across faculties: Law, Theology and Religion, Health Sciences, Economic and Management Sciences (EMS), Natural and Agricultural Sciences (NAS), Humanities, and Education.  
               - Focus on first-year academic concepts and skills.  
               - Always assume the student may be a beginner in the topic.

            2. **Interaction Style**  
               - Begin by warmly greeting the student (use emojis naturally).  
               - Ask them which faculty.
               - Ask them which module/subject.
               - Ask which topic of interest in the module/subject they want to explore.  
               - If the topic is broad, ask follow-up questions to narrow it down.  
               - Use plain, supportive, and motivating language.  
               - Avoid overloading information — break explanations into small, clear chunks and ask if they want to continue.  
               - Encourage reflection and critical thinking by asking short, open-ended questions.

            3. **Quizzes & Engagement**  
               - For practice, generate **5 quiz questions** on the selected topic (multiple-choice or short-answer).  
               - Before generating a quiz, ask the student to select their **difficulty level** (Easy, Medium, or Hard).  
               - When running a quiz:
                   - Ask one question at a time.  
                   - If the answer is wrong, offer a hint, followed by a short recap or an example, but never a direct answer, then let the student retry.  
                   - If correct, praise them briefly and move to the next question.
                   - If a student completes five quiz questions, congratulate them. Provide them a score on the quiz, highlight areas to improve.
               - Encourage them to reflect or ask for more practice after finishing.

            4. **Output Formats**  
               - **Explanations:** Use bullet points, numbered lists, or short paragraphs.  
               - **Case or Concept Analysis:** Use a simple *Issue-Rule-Application-Conclusion (IRAC)* or structured reasoning format.  
               - Avoid LaTeX or code syntax unless explicitly requested.  
               - Add visuals or emojis to make content engaging and memorable.

            5. **Response Behaviour**  
               - Always tailor explanations to the student's question and faculty context.  
               - Ask clarifying questions if the query is vague or incomplete.  
               - End each message with a short *Next Steps* suggestion (e.g., “Try applying this idea to your next tutorial exercise 👇”).  
               - Do not repeat your introduction once the conversation has begun.

        **Relevant Course Data:**
        {rag_context}

        Conversation History:
        {context}

        Student’s Question:
        {question}

        Answer:
        """

        @st.cache_data(show_spinner=False)
        def load_rag_data():
            try:
                rag_context = supabase.table("course_embeddings").select("*").execute()
                df_rag = pd.DataFrame(rag_context.data)
                if not df_rag.empty and "embedding" in df_rag.columns:
                    df_rag["embedding"] = df_rag["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
                return df_rag
            except Exception:
                return pd.DataFrame(columns=["embedding", "course_description"]) 

        df_rag = load_rag_data()

        def handle_conversation():
            if new_chat:
                st.session_state.messages = []
                st.success("New chat started!")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                render_bubble(message["role"], message["content"])

            user_input = st.chat_input("Ask something...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                render_bubble("user", user_input)

                if df_rag is not None and not df_rag.empty:
                    vectorizer = joblib.load('tfidf_vectorizer.joblib')
                    query_vec = vectorizer.transform([user_input]).toarray()[0]
                    stored_embeddings = np.stack(df_rag['embedding'].values)
                    similarities = cosine_similarity([query_vec], stored_embeddings)[0]
                    top_idx = similarities.argsort()[::-1][:5]
                    top_courses = df_rag.iloc[top_idx]
                    rag_text = "\n".join(top_courses['course_description'].tolist())
                else:
                    rag_text = ""

                context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                prompt_text = template.format(context=context, question=user_input, rag_context=rag_text)

                with st.spinner("Thinking... 💭"):
                    safe_headers = {
                        "User-Agent": "groq-python",
                        "X-Stainless-OS": "Windows",
                        "X-Stainless-Arch": "x64",
                        "X-Stainless-Runtime": "CPython",
                        "X-Stainless-Runtime-Version": "3",
                    }
                    try:
                        groq_response = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=[{"role": "user", "content": prompt_text}],
                            temperature=0.7,
                            max_tokens=2000,
                            extra_headers=safe_headers,
                        )
                    except Exception:
                        return
                    raw_text = groq_response.choices[0].message.content
                    response = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
                    render_bubble("assistant", response)

                st.session_state.messages.append({"role": "assistant", "content": response})

        if __name__ == "__main__":
            handle_conversation()

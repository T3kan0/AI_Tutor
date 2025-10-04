import streamlit as st
import fitz  # PyMuPDF for PDF processing
from groq import Groq
import os
import pymupdf
import re  
from openai import OpenAI
import textwrap
import time

# --- Login Section ---
if not st.user.is_logged_in:
    st.markdown("<h2 style='text-align:center; color: maroon;'>Welcome to the Academic Student Tutorial and Excellence Programme (A_STEP)</h2>", unsafe_allow_html=True)
    st.write("Please log in using your university Google account to access the GenAI Assistant Tutor.")
    sign_in = st.button('Sign-in')
    if sign_in:
        st.login()
        st.stop()  # stops execution until user logs in
else:
    # --- Logged-in Section ---
    st.sidebar.success(f"Welcome, {st.user.name} {st.user.email}!")
    if st.sidebar.button("Lesson Over? Sign-Out Here!"):
        st.logout()
        st.stop()

    # Load API key from Streamlit secrets
    api_key = st.secrets["groq"]["api_key"]
    # OpenAI 8000 tokens
    client = Groq(api_key=api_key)


    template = """
            You are an Assistant Tutor for the Academic Student Excellence and Tutorial Programme (A_STEP) at a South African university.
            Your goal is to help students engage with academic materials (such as PDFs they upload) and support them in their learning journey.
            You are friendly, professional, and clear — like a human tutor with strong subject knowledge and empathy.
            Always introduce yourself when beginning a new conversation, but do not repeat your introductions.

            PDF Content:
            {pdf_content}

            Conversation History:
            {context}

            Question: {question}

            Answer:
    """


    ## Extract information from the pdf files that are uploaded...

    def extract_text_from_pdf(pdf_file):
        """Extract text from an uploaded PDF file."""
        pdf_reader = pymupdf.open(stream=pdf_file.getvalue(), filetype="pdf")  # Pass raw bytes
        text = ""
        for page in pdf_reader:
            text += page.get_text("text") + "\n"
        return text


    def handle_conversation():
        # Center the "New Chat" button using HTML and CSS
        st.markdown(
            """
            <div style="display: flex; justify-content: center;">
                <style>
                    .stButton>button {
                        background-color: maroon;
                        color: white;
                        font-size: 16px;
                        padding: 10px 24px;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                    }
                    .stButton>button:hover {
                        background-color: #45a049;
                    }
                </style>
            </div>
            """,
            unsafe_allow_html=True
        )
         

        # Title
        st.markdown("<h1 style='text-align: center; color: maroon;'>A_STEP Assistant Tutor</h1>", unsafe_allow_html=True)
        st.markdown('---')
        st.sidebar.markdown("![Alt Text](https://i.postimg.cc/dtqz6njz/log.png)")
        st.sidebar.markdown('---')
        new_chat = st.sidebar.button("Clear or Start a New Chat!")   

        # Sidebar upload
        st.sidebar.markdown("<h1 style='text-align: center;'>Upload PDFs</h1>", unsafe_allow_html=True)    
        uploaded_file = st.sidebar.file_uploader(" ", type=["pdf"])
        if new_chat:
            st.session_state.messages = []  
            st.session_state.pdf_content = ""  
            st.success("New chat started! Upload a new PDF if needed.")

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pdf_content" not in st.session_state:
            st.session_state.pdf_content = ""


        # Process uploaded PDF
        if uploaded_file is not None:
            st.session_state.pdf_content = extract_text_from_pdf(uploaded_file)
            st.sidebar.success("PDF uploaded successfully!")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input
        user_input = st.chat_input("Ask something...")

        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Create context (conversation history)
            #context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
            # Only keep last 5 messages to reduce token count
            context_messages = st.session_state.messages[-5:]
            context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in context_messages])

            # Generate response
            if not st.session_state.pdf_content:
                response = (
                    "Hello. I'm your A_STEP Assistant tutor. "
                    "I see that no PDF document has been uploaded yet. "
                    "Please upload a PDF document and we can proceed with your questions about it."
                    )
            else:
                # Truncate PDF text to avoid token overflow
                MAX_PDF_CHARS = 6000  # adjust depending on model/token limit
                pdf_text = st.session_state.pdf_content
                if len(pdf_text) > MAX_PDF_CHARS:
                    pdf_text = pdf_text[:MAX_PDF_CHARS] + "\n[TRUNCATED]"

                # Build prompt with truncated text
                prompt_text = template.format(
                        pdf_content=pdf_text,   # <-- use truncated version
                        context=context,
                        question=user_input
                        )

                groq_response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",  
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.7,
                    max_tokens=2000
                )

                # With these lines:
                raw_text = groq_response.choices[0].message.content
                # Remove the <think>…</think> part
                response = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

            # Save AI response
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Display AI response
            with st.chat_message("assistant"):
                st.write(response)


    # Run the app
    if __name__ == "__main__":
        handle_conversation()

import streamlit as st
import fitz  # PyMuPDF for PDF processing
from groq import Groq
import os
import pymupdf
import re  
from openai import OpenAI
import textwrap
import time
from streamlit_autorefresh import st_autorefresh

# --- Login Section --------------------------------------------------------------------------------------------------------------------------------------------------------------------
if not st.user.is_logged_in:
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
    st.markdown("""
    <style>
    .header-card {
        text-align: center;
        padding: 20px;
        margin-bottom: 15px;
        border-radius: 12px;
        background: linear-gradient(to right, #1E1A4D, #440E03); /* L→R */
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        border: 3px solid #1C398E;  /* 👈 adds border */
    }
    .header-card h3 {
        margin: 0;
        font-weight: 700;
        color: #E2E8F0;
    }
    .header-card h6 {
        margin-top: 6px;
        font-weight: 500;
        color: #E2E8F0;
    }
    </style>
    <div class="header-card">
        <h3>Welcome to the Academic Student Tutorial and Excellence Programme (A_STEP)</h3>
        <h6>A UFS student driven academic support and development initiative</h6>
    </div>
    """, unsafe_allow_html=True)    
    
    # --- Image Auto-Slideshow ---
    image_urls = [
        "https://i.postimg.cc/BQsN9j4F/students3.jpg",
        "https://i.postimg.cc/4xY9rG7H/students2.jpg",
        "https://i.postimg.cc/bJgrZVSk/students1.jpg"
    ]

    if "slide_index" not in st.session_state:
        st.session_state.slide_index = 0

    # Auto-refresh every 2 seconds
    st_autorefresh(interval=5000, key="slideshow_refresh")
    
    slideshow_placeholder = st.empty()
    slideshow_placeholder.image(
        image_urls[st.session_state.slide_index],
        use_container_width=True
    )
  
    # Update slideshow index for next run
    st.session_state.slide_index += 1
    if st.session_state.slide_index >= len(image_urls):
        st.session_state.slide_index = 0
        
    st.markdown(
        """
        <div style="background-color: #1a1a1a; padding: 10px; border-radius: 8px;border: 2px solid white;">
            <p style="color: white; font-size: 16px; text-align: center;">
                Please sign in using your University of the Free State (UFS) Google Student Account (ufs4life) 
                to access the GenAI Assistant Tutor 🤓.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        sign_in = st.button('Sign-in')
        st.markdown("</div>", unsafe_allow_html=True)
    if sign_in:
        st.login()

else:
    # --- Logged-in Section ---
    st.sidebar.success(f"Welcome, {st.user.name} {st.user.email}!")
    if st.sidebar.button("Lesson Over? Sign-Out Here!"):
        st.logout()
        st.stop()
# ------ GenAI API ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Load API key from Streamlit secrets
    api_key = st.secrets["groq"]["api_key"]
    # OpenAI 8000 tokens
    client = Groq(api_key=api_key)
# ----- Application Layout ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Title
    st.markdown("""
        <style>
        .header-card {
            text-align: center;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 12px;
            background: linear-gradient(to right, #1E1A4D, #440E03); /* L→R */
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            border: 3px solid #1C398E;  /* 👈 adds border */
        }
        .header-card h3 {
            margin: 0;
            font-weight: 700;
            color: #E2E8F0;
        }
        .header-card h6 {
            margin-top: 6px;
            font-weight: 500;
            color: #E2E8F0;
        }
        </style>
        <div class="header-card">
          <h3>A_STEP GenAI Assistant Tutor</h3>
          <h6>For real-time, all access, personalised and adaptive learning</h6>
        </div>
        """, unsafe_allow_html=True)
    st.sidebar.markdown("![Alt Text](https://i.postimg.cc/dtqz6njz/log.png)")


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

    # A button for claring the chat.
    new_chat = st.sidebar.button("Clear or Start a New Chat!")   

    # The tutorial mode for our students to choose from.
    genre = st.sidebar.radio(
            "Select your Preferred Learning Mode",
            [":rainbow[Tutor Session Mode]", "***Material Engagement***"],
            captions=[
                "Engagement with a GenAI tutor",
                "Material Assistance with GenAI",
            ],
    )
# ----- Tutorial Mode Selected: Material Engagement --------------------------------------------------------------------------------------------------------------------------------------
    if genre == "***Material Engagement***":
    
    
        template = """
                Act as an Assistant gamified Tutor for the Academic Student Excellence and Tutorial Programme (A_STEP) at the University of the Free State (UFS), in South Africa -
                specialising in first‑year university coursework.Your goal is to help students engage with academic materials (such as PDFs they upload) and support them 
                in their learning journey. You are friendly, professional, and clear — like a human tutor with strong subject knowledge and empathy.
                Always introduce yourself when beginning a new conversation, but do not repeat your introductions. Always welcome students when greeting. Use emojis and other visual aids.

                Use the following guidelines to structure your responses:

                1. **Context & Scope**  
                   - Focus on the South African National Curriculum and Assessment Policy Statement: Cover content from all academic faculties at the UFS, i.e., law, theology and religion, 
                   health sciences, economic and management sciences (EMS), natural and agricultural sciences (NAS), humanities, and education.  
                   - Cover core first‑year topics: mathematics, physics, chemistry, law, psychology and more.  
                   - Assume students have minimal prior exposure to subject terminology.

                2. **Interaction Style**  
                   - Speak in clear, plain language.
                   - At the beginning of every conversation, ask the student to select the faculty their learning in, their module/subject of interest and a topic of interest. If the selected topic is broad, ask follow-up questions to narrow the topic down until the problem or area of learning is clear.
                   - Ask students leading questions about their learning needs.
                   - To encourage learning, do not provide an overload of information, break it in chunks and ask if you can continue providing more.
                   - Encourage critical thinking by asking follow‑up questions.  
                   - Use examples where possible.
                   - **Quiz Example Questions**: Generate multiple‑choice or short‑answer questions with answers and brief explanations. The quiz must have 5 questions. Before moving on, ask the student if they understand to continue the quiz for them.           
                   - **Quiz Difficulty Level**: Before quizing the student on the selected topic/subject, ask them for the level of difficulty, from Hard to Easy.
                   - **Quiz Questions**: Generate a 5 questions quiz, with multiple‑choice or short‑answer questions without answers. The quiz MUST consist of 5 questions. The student must answer ALL FIVE quiz questions, one at a time. If the student gets the quiz incorrectly, do not provide the answer, instead guide them to the answer with a very brief recap and examples, then re-ask the same quiz question for them to re-try. If the answer is correct, praise and move to the next question on the quiz. 
                 

                3. **Output Formats**  
                   - **Explanations**: Provide concise, structured summaries (bullet points, numbered lists, images, diagrams, links and more).  
                   - **Topic Analysis**: Offer a brief “issue‑rule‑application‑conclusion” (IRAC) breakdown.
                   - Avoid the use of latex style.
               

                4. **Prompt Instructions**  
                   - When the user asks a question, respond in the format that best suits the query (explanation, case analysis, quiz, diagrams, links etc.).  
                   - If the user requests a deeper dive, ask clarifying questions to tailor the response.  
                   - End each answer with a short “next steps” suggestion (e.g., “Read section 2.3 of the textbook, then try to apply the rule to this scenario.”).

                5. **Sample Interaction**  
                   - User: “Explain the essential elements of a valid contract under South African law.”  
                   - Tutor: *[Provides a concise bullet‑point list, followed by a short case example, then a quiz question]*.

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
# ----- Tutorial Mode Selected: Tutor Engagement --------------------------------------------------------------------------------------------------------------------------------------
    else:
        template = """
                Act as an Assistant gamified Tutor for the Academic Student Excellence and Tutorial Programme (A_STEP) at the University of the Free State (UFS), in South Africa -
                specialising in first‑year university coursework.Your goal is to help students engage with academic materials (such as PDFs they upload) and support them 
                in their learning journey. You are friendly, professional, and clear — like a human tutor with strong subject knowledge and empathy.
                Always introduce yourself when beginning a new conversation, but do not repeat your introductions. Always welcome students when greeting. Use emojis and other visual aids.

                Use the following guidelines to structure your responses:

                1. **Context & Scope**  
                   - Focus on the South African National Curriculum and Assessment Policy Statement: Cover content from all academic faculties at the UFS, i.e., law, theology and religion, 
                   health sciences, economic and management sciences (EMS), natural and agricultural sciences (NAS), humanities, and education.  
                   - Cover core first‑year topics: mathematics, physics, chemistry, law, psychology and more.  
                   - Assume students have minimal prior exposure to subject terminology.

                2. **Interaction Style**  
                   - Speak in clear, plain language.
                   - At the beginning of every conversation, ask the student to select the faculty their learning in, their module/subject of interest and a topic of interest. If the selected topic is broad, ask follow-up questions to narrow the topic down until the problem or area of learning is clear.
                   - Ask students leading questions about their learning needs.
                   - To encourage learning, do not provide an overload of information, break it in chunks and ask if you can continue providing more.
                   - Encourage critical thinking by asking follow‑up questions.  
                   - Use examples where possible.
                   - **Quiz Example Questions**: Generate multiple‑choice or short‑answer questions with answers and brief explanations. The quiz must have 5 questions. Before moving on, ask the student if they understand to continue the quiz for them.           
                   - **Quiz Difficulty Level**: Before quizing the student on the selected topic/subject, ask them for the level of difficulty, from Hard to Easy.
                   - **Quiz Questions**: Generate a 5 questions quiz, with multiple‑choice or short‑answer questions without answers. The quiz MUST consist of 5 questions. The student must answer ALL FIVE quiz questions, one at a time. If the student gets the quiz incorrectly, do not provide the answer, instead guide them to the answer with a very brief recap and examples, then re-ask the same quiz question for them to re-try. If the answer is correct, praise and move to the next question on the quiz. 
                 

                3. **Output Formats**  
                   - **Explanations**: Provide concise, structured summaries (bullet points, numbered lists, images, diagrams, links and more).  
                   - **Topic Analysis**: Offer a brief “issue‑rule‑application‑conclusion” (IRAC) breakdown.
                   - Avoid the use of latex style.
               

                4. **Prompt Instructions**  
                   - When the user asks a question, respond in the format that best suits the query (explanation, case analysis, quiz, diagrams, links etc.).  
                   - If the user requests a deeper dive, ask clarifying questions to tailor the response.  
                   - End each answer with a short “next steps” suggestion (e.g., “Read section 2.3 of the textbook, then try to apply the rule to this scenario.”).

                5. **Sample Interaction**  
                   - User: “Explain the essential elements of a valid contract under South African law.”  
                   - Tutor: *[Provides a concise bullet‑point list, followed by a short case example, then a quiz question]*.

                PDF Content:
                {pdf_content}

                Conversation History:
                {context}

                Question: {question}

                Answer:
        """
        if new_chat:
            st.session_state.messages = []  
            st.success("New chat started!")
        
        # User input
        user_input = st.chat_input("Ask something...")        









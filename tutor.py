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

# --- Login Section --------------------------------------------------------------------------------------------------------------------------------------------------------------------

def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)

if not st.user.is_logged_in:
    #login_screen()

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
    #login_screen()
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        #sign_in = st.button('Sign-in')
        login_screen()
        st.markdown("</div>", unsafe_allow_html=True)

else:
    #st.user
    st.sidebar.success(f"Welcome, {st.user.name} {st.user.email}!")
    st.sidebar.button("Log out", on_click=st.logout)
# --- Logged-in Section --------------------------------------------------------

# ------ GenAI API ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Load API key from Streamlit secrets
    api_key = st.secrets["groq"]["api_key"]
    # OpenAI 8000 tokens
    client = Groq(api_key=api_key)
    SUPABASE_URL1 = st.secrets["vectors"]["SUPABASE_URL1"]
    SUPABASE_KEY1 = st.secrets["vectors"]["SUPABASE_KEY1"]

    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL1, SUPABASE_KEY1)
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
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Create context (conversation history)
                #context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
                # Only keep last 5 messages to reduce token count
                context_messages = st.session_state.messages[-5:]
                context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in context_messages])

                # Generate response
                if not st.session_state.pdf_content:
                    response = (
                        "Hi there 👋. Welcome to the ***Material Engagement*** Tutorial Session. I'm your A_STEP Assistant tutor ✨ . "
                        "I see that no PDF document has been uploaded yet 🤷 . "
                        "Use the upload button to navigate to a PDF document 📖, then we can proceed with your questions about it, otherwise you can switch to the ***Tutor Session Mode*** to Chat with a GenAI Tutor 🧑‍🏫."
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
                   - NEVER PROVIDE A DIRECT ANSWER TO THE STUDENT.
                   - If the answer is wrong, offer a hint, followed by a short recap, a hint or an example, then let the student retry. If the retry is correct, move on to the next question.
                   - If correct, praise them briefly and move to the next question.
                   - Guide the student to complete all five quiz questions.
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
        # --- pull your RAG data from the database ---
        @st.cache_data(show_spinner=False)
        def load_rag_data():
        
            # Fetch all data from your table
            rag_context = supabase.table("course_embeddings").select("*").execute()
            # Convert to pandas DataFrame
            df_rag = pd.DataFrame(rag_context.data)

            # Convert string embeddings to list of floats
            df_rag["embedding"] = df_rag["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
            return df_rag
        
        df_rag = load_rag_data()
        # Quick sanity check preview
        #st.write("✅ Loaded RAG data:", df_rag.shape)
        #st.dataframe(df_rag.head())
        
        def handle_conversation():
            if new_chat:
                st.session_state.messages = []  
                st.success("New chat started!")

            # Initialize session state
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
            # User input
            user_input = st.chat_input("Ask something...")        

            if user_input:
                # Append user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)

                # Load pre-fitted vectorizer
                vectorizer = joblib.load('tfidf_vectorizer.joblib')

                # Embed the user query
                query_vec = vectorizer.transform([user_input]).toarray()[0]  # same TF-IDF space
                
                # Compute similarity against stored embeddings
                stored_embeddings = np.stack(df_rag['embedding'].values)  # already in app
                similarities = cosine_similarity([query_vec], stored_embeddings)[0]

                top_idx = similarities.argsort()[::-1][:5]
                top_courses = df_rag.iloc[top_idx]                
                # Combine top course descriptions as RAG context
                rag_text = "\n".join(top_courses['course_description'].tolist())
                
                # Prepare conversation context
                context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])

                # Extend your prompt with RAG context
                prompt_text = template.format(context=context, question=user_input, rag_context=rag_text) + "\n\nRelevant courses:\n" + rag_text
                
                # Format prompt
                #prompt_text = template.format(context=context, question=user_input)
                
                # Generate tutor response (OpenAI Model)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking... 💭"):
                        groq_response = client.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=[{"role": "user", "content": prompt_text}],
                            temperature=0.7,
                            max_tokens=2000
                        )
  
                        # Extract and clean response
                        raw_text = groq_response.choices[0].message.content
                        response = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

                        st.write(response)
                
                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": response})

        
        # Run the app
        if __name__ == "__main__":
            handle_conversation()







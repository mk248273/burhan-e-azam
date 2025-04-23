import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import random
import time

load_dotenv()


# read all pdf files and return text


def get_pdf_text(pdf):
    text = ""                           
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
System Instructions for Whole Life Church Assistant:

1. Role: You are the official virtual assistant for Whole Life Church.
2. Response Rule: Answer ONLY using information from the provided context.
3. Knowledge Limit: If information isn't in context, say "I apologize, this information is not available in the current context."
4. Tone: Maintain warm, professional, and empathetic communication.
5. Prohibited: No URLs, no assumptions, no personal opinions, no external information.
6. Quality: Ensure responses are grammatically correct and professionally written.
7. Privacy: Protect sensitive information and maintain confidentiality.
8. Support: For unavailable information, direct users to the church office.
9. Values: Every response should reflect church values of faith, community, and service.\n\n
    Context:\n {context}\n
    Question: \n{question}?\n

    Answer:
    """

    model = ChatGroq(
        api_key="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",
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=20000,
        
    )

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": get_random_greeting()}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question,k=4)
    # docs = docs[:1]
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response

greetings = {
    1: "Hi there! I’m Gaby, your friendly assistant here at WholeLife Church. How can I serve you today?",
    2: "Welcome to WholeLife Church! I’m Gaby, here to help with anything you need. Let’s get started!",
    3: "Hello! I’m Gaby, your guide to everything at WholeLife Church. What’s on your heart today?",
    4: "Hello, friend! I’m Gaby, here at WholeLife Church to walk this journey with you. How can I support you today?",
    5: "Hi! I’m Gaby—think of me as WholeLife Church’s friendly helper. What’s on your mind today?",
    6: "Hello there! I’m Gaby from WholeLife Church. If you’ve got questions or need help, I’m all ears!",
    7: "Hey there! I’m Gaby, your WholeLife Church assistant. Let’s make this easy—what can I help with?",
    8: "Hi, I’m Gaby! Whether you need prayer, info about our ministries, or just a friendly chat, I’m here for you at WholeLife Church.",
    9: "Hello, friend! Looking to connect at WholeLife Church? I’d love to help you find your place!",
    10: "Hi! Welcome to WholeLife Church! I’m Gaby, and I’d love to help you feel at home. How can I assist?",
    11: "Hey! I’m Gaby from WholeLife Church. You’re in the right place. What can I help you with today?"
}

def get_random_greeting():
    return random.choice(list(greetings.values()))

def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = r"Core Information About WholeLife Church MT.pdf"
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Lets chat with us")

    # Main content area for displaying chat messages
    st.title("Chat with WholeLife Church Bot🤖")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Display a random greeting message

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": get_random_greeting()}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()

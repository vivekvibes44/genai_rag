
import os
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.header("My Chatbot")

# Upload pdf
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(
        "Upload a PDF file and start asking questions",
        type="pdf",
    )

# Extract contents from the PDF
if file is not None:
    text = ""

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    # Store embeddings in vector database
    if chunks:
        vector_store = FAISS.from_texts(chunks, embeddings)

        # Get user question
        user_question = st.text_input("Type your question here")

        # Format retrieved documents
        def format_docs(docs):
            return "\n\n".join(
                [doc.page_content for doc in docs]
            )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # Define LLM
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=512,
            api_key=GROQ_API_KEY,
        )

        # Define the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant answering questions about a PDF document.\n\n"
                    "Guidelines:\n"
                    "1. Provide complete, well-explained answers using the context below.\n"
                    "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
                    "3. Include related information from the context to provide a full picture.\n"
                    "4. Only use information from the provided context; do not use outside knowledge.\n"
                    "5. Summarize long information, ideally in bullets where needed.\n"
                    "6. If the information is not in the context, say so politely.\n\n"
                    "Context:\n{context}",
                ),
                ("human", "{question}"),
            ]
        )

        # Define flow chain: Question -> retrieval -> llm -> response
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        if user_question:
            response = chain.invoke(user_question)
            st.write(response)

    else:
        st.warning(
            "No extractable text found in this PDF. "
        )

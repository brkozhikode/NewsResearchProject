from datetime import time

import streamlit as st
import os
import ssl
import time
#import pickle

from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)


st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context
embeddings = OpenAIEmbeddings()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    #split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # Save the vectorstore object locally
    vectorstore_openai.save_local("vectorstore")



query = main_placeholder.text_input("Question:")

if query:
        main_placeholder.write(f"Query Started Building...✅✅✅, {query}")
        vectorstore_local = FAISS.load_local("vectorstore", embeddings,   allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_local.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
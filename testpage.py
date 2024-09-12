import streamlit as st

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def askquestion(urls):
    loader = UnstructuredURLLoader(urls=urls)
    # main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split text
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    # main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    # main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    # time.sleep(2)

    # Save the FAISS index to a pickle file
    # with open(file_path, "wb") as f:
    #   pickle.dump(vectorstore_openai, f)
    with st.form(key='my_form'):
        query = st.text_input("Question:")
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.write(f"Query Started Building...✅✅✅, {query}")
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_openai.as_retriever())
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
        else:
            st.write(f"Query is empty or false., {query}")
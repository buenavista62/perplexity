import streamlit as st
from dotenv import load_dotenv

from llm import process_pdf, create_rag_chain

load_dotenv()

def main():
    st.title("PDF Question Answering App")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    chunk_size = st.number_input("Chunk Size", min_value=50, max_value=1000, value=200, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=100, value=30, step=10)

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = process_pdf(uploaded_file, chunk_size, chunk_overlap)
                st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
            st.success("PDF processed successfully!")

    if st.session_state.vectorstore is not None:
        query = st.text_input("Ask a question about the PDF:")

        if query:
            with st.spinner("Generating answer..."):
                response = st.session_state.rag_chain.invoke({"query": query})
                result = response['result']
                source_documents = response['source_documents']

            st.subheader("Answer:")
            st.write(result)

            st.subheader("Sources:")
            for doc in source_documents:
                st.write(f"Page {doc.metadata['page']}")
                st.write(f"Chunk: {doc.page_content}...")
                st.write("---")

if __name__ == "__main__":
    main()

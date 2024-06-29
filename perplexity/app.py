import streamlit as st
from dotenv import load_dotenv

from llm import process_pdf, create_rag_chain
from embed_models import models

load_dotenv()

def main():
    st.title("PDF Question Answering App")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "configured" not in st.session_state:
        st.session_state.configured = False

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    @st.experimental_dialog("Configuration")
    def config():
        embed_model = st.selectbox("Embedding Model", models.items(), index=0, format_func=lambda x: x[0])
        chunk_size = st.number_input("Chunk Size", min_value=50, max_value=1000, value=200, step=50)
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=100, value=30, step=10)
        n_index = st.number_input("Index Size", min_value=2, max_value=100, value=10, step=1)
        if st.button("Apply"):
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.n_index = n_index
            st.session_state.embed_model = embed_model
            st.session_state.configured = True
            st.rerun()


    

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = process_pdf(uploaded_file, chunk_size = st.session_state.chunk_size, 
                chunk_overlap = st.session_state.chunk_overlap, n_indexes=st.session_state.n_index,
                embed_model = st.session_state.embed_model[1])
                st.session_state.rag_chain = create_rag_chain(st.session_state.vectorstore)
            st.success("PDF processed successfully!")
    
    if st.button("configuration"):
        config()

    if st.session_state.configured:    
        st.write(f"Your current configuration: Chunk Size: {st.session_state.chunk_size}, "
        f"Chunk Overlap: {st.session_state.chunk_overlap}, Index Size: {st.session_state.n_index},"
        f"Embedding Model: {st.session_state.embed_model[0]}")

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

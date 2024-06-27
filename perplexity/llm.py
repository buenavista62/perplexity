import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Annoy
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.docstore import InMemoryDocstore
from langchain_community.chat_models import ChatPerplexity
from langchain.text_splitter import MarkdownTextSplitter

import os
import pymupdf4llm
from annoy import AnnoyIndex


def process_pdf(uploaded_file, chunk_size, chunk_overlap):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        # Write the uploaded file data to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Convert PDF to Markdown using the temporary file path
        md_text = pymupdf4llm.to_markdown(tmp_file_path, page_chunks=True)

        # Rest of your processing code...
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        documents = []

        for page in md_text:
            page_text = page["text"]
            metadata = page["metadata"]

            page_docs = splitter.create_documents([page_text], metadatas=[metadata])
            documents.extend(page_docs)

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/gtr-t5-large"
        )

        texts = [doc.page_content for doc in documents]
        embeddings = embedding_model.embed_documents(texts)

        embedding_dim = len(embeddings[0])
        annoy_index = AnnoyIndex(embedding_dim, "angular")

        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)

        annoy_index.build(20)  # 20 trees

        index_to_docstore_id = {i: str(i) for i in range(len(documents))}

        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})

        vectorstore = Annoy(
            embedding_function=embedding_model.embed_query,
            index=annoy_index,
            metric="angular",
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )

        return vectorstore

    finally:
        # Remove the temporary file
        os.unlink(tmp_file_path)


def create_rag_chain(vectorstore):
    # Create retriever
    retriever = vectorstore.as_retriever()

    # Create chat model
    chat = ChatPerplexity(
        temperature=0.3,
        pplx_api_key=os.getenv("PERPLEXITY_API_KEY"),
        model="llama-3-sonar-small-32k-chat",
    )

    # Create prompt template
    system = (
        "You are an artificial intelligence assistant. Use the following"
        "pieces of context to answer the user's question. If you don't know the"
        "answer, just say that you don't know, don't try to make up an answer."
    )
    human = "Context: {context}\n\nHuman: {question}\n\nAssistant:"

    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    # Create RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True,
    )

    return rag_chain

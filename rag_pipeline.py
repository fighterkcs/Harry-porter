import os
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader

# --------------------------
# 1️⃣ LOAD ALL PDFS
# --------------------------
def load_all_pdfs(data_folder):
    """Loads text using LangChain's PyPDFLoader from all PDFs."""
    documents = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file_name)
            print(f"📘 Loading: {file_name}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)

    print(f"✅ Loaded {len(documents)} pages from {data_folder}")
    combined_text = " ".join([doc.page_content for doc in documents])
    return combined_text


# --------------------------
# 2️⃣ CHUNKING
# --------------------------
def chunk_text(text, chunk_size=450):
    """Splits text into chunks for embedding."""
    return textwrap.wrap(text, chunk_size)


# --------------------------
# 3️⃣ CREATE VECTOR STORE
# --------------------------
def create_vector_store(chunks):
    """Encodes chunks into embeddings and stores them in FAISS."""
    print("⚙️ Creating embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = embed_model.encode(chunks)
    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(chunk_embeddings))
    print(f"✅ Added {len(chunks)} chunks to FAISS index.")
    return embed_model, index


# --------------------------
# 4️⃣ RETRIEVAL + ANSWERING
# --------------------------
def retrieve_and_answer(query, embed_model, index, chunks, qa_pipe, top_k=3):
    """Retrieves top chunks and generates an answer."""
    query_embedding = embed_model.encode([query])
    _, indices = index.search(np.array(query_embedding), top_k)
    retrieved_texts = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_texts)

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = qa_pipe(prompt, max_new_tokens=200)
    return result[0]["generated_text"]


# --------------------------
# 5️⃣ INITIALIZE PIPELINE (For Streamlit)
# --------------------------
def initialize_pipeline():
    """Loads PDFs, creates chunks, embeddings, and initializes model."""
    DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")

    print("📂 Reading all PDFs from:", DATA_FOLDER)
    document_text = load_all_pdfs(DATA_FOLDER)

    print("✂️ Chunking text...")
    chunks = chunk_text(document_text, chunk_size=450)

    embed_model, index = create_vector_store(chunks)
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-small")

    print("🤖 RAG pipeline ready!")
    return embed_model, index, chunks, qa_pipe
import streamlit as st
from rag_pipeline import initialize_pipeline, retrieve_and_answer

# Streamlit UI
st.set_page_config(page_title="Harry Potter RAG 🧙‍♂️", page_icon="📚")

st.title("🧙‍♂️ Harry Potter RAG Chatbot")
st.write("Ask any question from the 5 Harry Potter books (PDFs).")

# Initialize only once
@st.cache_resource
def load_pipeline():
    return initialize_pipeline()

embed_model, index, chunks, qa_pipe = load_pipeline()

# User input
query = st.text_input("🔍 Ask your question:")

# Compute similarity score for the top chunk
if query.strip():
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, 1)  # 1 = top result
    similarity_score = 1 / (1 + D[0][0]) if D[0][0] > 0 else 1.0  # Convert distance to similarity (simple heuristic)
else:
    similarity_score = 0

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... 🧠"):
            if similarity_score < 0.4:  # was 0.65
                st.write("Your query is out of context for the documents.")
            else:
                answer = retrieve_and_answer(query, embed_model, index, chunks, qa_pipe)
                st.write(answer)

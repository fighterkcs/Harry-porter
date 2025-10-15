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

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question!")
    else:
        with st.spinner("Thinking... 🧠"):
            answer = retrieve_and_answer(query, embed_model, index, chunks, qa_pipe)
            st.success(answer)
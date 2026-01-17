import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import pipeline
import streamlit as st
from rag_pipeline import initialize_pipeline, retrieve_and_answer

load_dotenv()

# Streamlit UI
st.set_page_config(page_title="YOGA ", page_icon="📚")

st.title("HEALTH IS WEALTH")


# Initialize only once
@st.cache_resource
def load_pipeline():
    return initialize_pipeline()

@st.cache_resource
def load_fallback_models():
    try:
        local_generator = pipeline("text_generation", model="gpt2")
    except Exception as e:
        print("failed to load GPT-2 locally:", e)
        local_generator = None
    
    hf_token = os.getenv("hf_token1") or os.getenv("hf_token2")
    client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
    return local_generator, client

local_generator, client = load_fallback_models()

def generate_text_fallback(prompt):
  try:
    print("Trying GPT 2 locally...")
    if local_generator:
      result=local_generator(prompt,max_length=50)
      return f"GPT-2 (local):\n{result[0]['generated_text']}"
    else:
        raise RuntimeError("GPT-2 not available")
  except Exception as e:
          print("Local GPT-2 failed . Falling back to hugging face Inference API...")
          try:
            messages=[
                {"role":"system","content":"you are a helpful assistant"},
                {"role":"user","content":prompt}
            ]
            response=client.chat_completion(messages=messages,max_tokens=50)
            result=response.choices[0].message.content.strip()
            return f"Hugging face API:\n{result}"
          except Exception as api_error:
            return f"Both methods Failed.\nerror:\n{traceback.format_exc()}"

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
                st.write(generate_text_fallback(query))
            else:
                answer = retrieve_and_answer(query, embed_model, index, chunks, qa_pipe)
                st.write(answer)

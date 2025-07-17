import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data files (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# --- CONFIG ---
PDF_DIR = '/home/rubesh/Desktop/sweta/Mtech_internship/another_repo/papers/multi_camera_tracking'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500  # words
TOP_K = 5

# --- PDF TEXT EXTRACTION ---
def extract_texts(pdf_folder):
    texts = []
    for fname in os.listdir(pdf_folder):
        if fname.endswith('.pdf'):
            with pdfplumber.open(os.path.join(pdf_folder, fname)) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                texts.append(text)
    return texts

# --- CHUNKING ---
def chunk_texts(texts):
    chunks = []
    for text in texts:
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para.split()) > 30:  # skip very short paragraphs
                chunks.append(para)
    return chunks

# --- EMBEDDING & INDEXING ---
@st.cache_resource
def build_index():
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = extract_texts(PDF_DIR)
    chunks = chunk_texts(texts)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, chunks, index

# --- RETRIEVAL ---
def retrieve(query, model, chunks, index, top_k=TOP_K):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]]

@st.cache_resource
def get_llm():
    # Use a general-purpose, open-source LLM for zero-shot Q&A
    # Example: FLAN-T5 Large (open-access, good for Q&A and summarization)
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_new_tokens=512,  # Increased for longer responses
    )

llm = get_llm()

def clean_context(text):
    # Remove hyphenation at line breaks
    text = re.sub(r'-\s*\n\s*', '', text)
    # Replace multiple newlines and spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def llm_answer(context, question):
    # Clean context to improve LLM output
    context = clean_context(context)
    max_context_words = 400  # Increased for richer context
    context_words = context.split()
    if len(context_words) > max_context_words:
        context = " ".join(context_words[:max_context_words])
    prompt = (
        "You are an expert in computer vision and multi-camera tracking. "
        "Given the following context from several research papers, compare and summarize the main methods and approaches used to tackle the problem of Multi-Camera Vehicle Tracking (MCVT). "
        "Highlight differences, innovations, and any unique techniques across the papers. "
        "Provide a detailed and structured summary.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    result = llm(prompt)
    return result[0]['generated_text'] if isinstance(result, list) else result

def extract_query_keywords(query):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(query.lower())
    keywords = [w for w in words if w.isalpha() and w not in stop_words]
    # Add important phrases manually
    if "single camera" in query.lower():
        keywords.append("single camera")
        keywords.append("single-camera")
        keywords.append("single_camera")
    if "multi camera" in query.lower():
        keywords.append("multi camera")
        keywords.append("multi-camera")
        keywords.append("multi_camera")
    return keywords

def filter_chunks(chunks, query=None):
    method_keywords = [
        "method", "approach", "solution", "proposed", "algorithm", "model", "tackle", "address", "technique"
    ]
    query_keywords = extract_query_keywords(query) if query else []
    filtered = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in method_keywords + query_keywords) and "abstract" in chunk_lower:
            filtered.append(chunk)
    # Fallback: if no chunk matches, use original filter
    if not filtered:
        for chunk in chunks:
            if "abstract" in chunk.lower():
                filtered.append(chunk)
    # Final fallback: if still empty, use all chunks
    return filtered if filtered else chunks

def deduplicate_chunks(chunks):
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        cleaned = clean_context(chunk)
        if cleaned not in seen:
            seen.add(cleaned)
            unique_chunks.append(chunk)
    return unique_chunks

def extract_links(chunk):
    # Improved regex to extract only valid URLs, DOIs, arXiv links
    urls = re.findall(r'(https?://[\w\-\./\?\=\#\%]+|doi:[\w\./]+|arxiv:[\w\./]+)', chunk)
    # Remove duplicates and filter out publisher/terms links
    unique_urls = set()
    valid_urls = []
    for url in urls:
        if url.lower().startswith(('http://', 'https://')) or url.lower().startswith(('doi:', 'arxiv:')):
            # Skip links that are just publisher terms or copyright
            if 'springernature' in url or 'terms' in url or 'policy' in url:
                continue
            if url not in unique_urls:
                unique_urls.add(url)
                valid_urls.append(url)
    return valid_urls

# --- UI ---
st.title('Multi-Camera Tracking Papers RAG System')
st.write('Ask a question about multi-camera tracking papers:')

model, chunks, index = build_index()

query = st.text_input('Enter your question:')

st.subheader('LLM Answer')
if query:
    # Use the top 15 relevant context chunks for the LLM, then filter and deduplicate them
    context_chunks = retrieve(query, model, chunks, index, top_k=15)
    filtered_chunks = filter_chunks(context_chunks, query=query)
    deduped_chunks = deduplicate_chunks(filtered_chunks)
    # Concatenate and clean all deduplicated chunks, then truncate to 400 words
    combined_context = " ".join([clean_context(chunk) for chunk in deduped_chunks])
    max_context_words = 400
    context_words = combined_context.split()
    if len(context_words) > max_context_words:
        combined_context = " ".join(context_words[:max_context_words])
    answer = llm_answer(combined_context, query)
    st.markdown(answer)
    st.write('---')
    st.write('Top relevant context:')
    for i, chunk in enumerate(deduped_chunks):
        st.markdown(f'**Chunk {i+1}:** {clean_context(chunk)[:500]}...')
        links = extract_links(chunk)
        if links:
            for link in links:
                st.markdown(f'🔗 [Paper Link]({link}) ({link})')
else:
    st.markdown("")
    st.write('---')
    st.write('Top relevant contexts:') # context from the RAG systems 
    st.markdown("")

st.write('---')
st.write('Upload new papers to /papers/multi_camera_tracking and restart to update.')

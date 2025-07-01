import streamlit as st
import fitz  # PyMuPDF
import os
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4f46e5;
        background: #f0f9ff;
    }
    
    /* File list styling */
    .file-item {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4f46e5;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .file-info {
        display: flex;
        flex-direction: column;
    }
    
    .file-name {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    
    .file-size {
        font-size: 0.85rem;
        color: #6b7280;
    }
    
    /* Question input styling */
    .question-container {
        background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
    }
    
    /* Answer styling */
    .answer-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }
    
    .source-info {
        background: #e0f2fe;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        border-left: 4px solid #0284c7;
    }
    
    .source-label {
        font-weight: 600;
        color: #0284c7;
        margin-bottom: 0.5rem;
    }
    
    /* Status message styling */
    .status-success {
        background: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #a7f3d0;
        margin-bottom: 1rem;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #fca5a5;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* PDF preview styling */
    .pdf-preview {
        text-align: center;
        margin-top: 1rem;
    }
    
    .pdf-page {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        max-width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'texts' not in st.session_state:
    st.session_state.texts = []
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# Configuration
DOCS_FOLDER = "docs"
INDEX_FILE = "myindex.faiss"
TEXT_FILE = "mytext.json"
API_KEY = "YOUR-API-KEY"  # Replace with your API key

# Create docs folder if it doesn't exist
os.makedirs(DOCS_FOLDER, exist_ok=True)

# Initialize models and index
@st.cache_resource
def load_models():
    """Load SentenceTransformer model and initialize LLM"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load sentence transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=API_KEY
    )
    
    return model, llm

def load_index():
    """Load FAISS index and texts"""
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(384)
    
    if os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "r") as f:
            texts = json.load(f)
    else:
        texts = []
    
    return index, texts

def save_index_and_texts(index, texts):
    """Save FAISS index and texts to files"""
    faiss.write_index(index, INDEX_FILE)
    with open(TEXT_FILE, "w") as f:
        json.dump(texts, f)

def load_and_chunk_pdf(pdf_path):
    """Extract text from PDF and create chunks"""
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            chunks.append({
                "filename": os.path.basename(pdf_path),
                "text": text,
                "page_no": page_num
            })
    doc.close()
    return chunks

def add_text_to_index(new_items, model, index, texts):
    """Add new text chunks to the FAISS index"""
    if not new_items:
        return index, texts
    
    raw_texts = [item["text"] for item in new_items]
    embeddings = model.encode(raw_texts, normalize_embeddings=True).astype("float32")
    
    index.add(embeddings)
    texts.extend(new_items)
    
    return index, texts

def retrieve(query, model, index, texts, top_k=5):
    """Retrieve relevant chunks for a query"""
    if len(texts) == 0:
        return []
    
    query_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_emb, top_k)
    return [texts[i] for i in I[0] if i != -1 and i < len(texts)]

def json_to_obj(json_str: str) -> dict:
    """Clean and parse JSON response from LLM"""
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_str.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"answer": "Error parsing response", "filename": None, "page_no": None}

def rag_answer(question, model, index, texts, llm):
    """Generate answer using RAG approach"""
    chunks = retrieve(question, model, index, texts)
    if not chunks:
        return {"answer": "No relevant information found. Please upload some PDF documents first.", "filename": None, "page_no": None}
    
    context = json.dumps(chunks)
    prompt_template = """
You are a helpful and intelligent study assistant. You are given a list of JSON objects as context, each representing extracted text from a PDF.

Each object contains:
- 'text': the actual content
- 'filename': the PDF file name
- 'page_no': the page number

Your task is to answer the student's question based **primarily** on the 'text' fields in the context. You may reason and infer the answer if the exact wording is not available, as long as your answer is clearly supported by the content.

# Context:
{context}

# Question:
{question}

# Instructions:
- Use only the 'text' field from the context entries for answering the question, but include the most relevant 'filename' and 'page_no' you used.
- Even if the original text is technical or unclear, rewrite the answer in a simple, **student-friendly** way that is easy to understand.
- You can use **Markdown formatting** (headings, bullet points, code blocks, tables, etc.) to make the answer more readable and structured.
- You **may infer** or **summarize** answers from the content to help students understand, even if the answer is not a perfect match.
- Avoid saying you don't know unless the question is entirely unrelated to the context.
- Return a JSON object with:
  - "answer": your helpful, clear, Markdown-formatted answer
  - "filename": the filename of the most relevant entry you used
  - "page_no": the corresponding page number

- If the context contains **nothing relevant at all** to the question, return:
  ```json
  {{
    "answer": "I'm sorry, but I don't have enough information to answer that.",
    "filename": null,
    "page_no": null
  }}
  ```
Respond ONLY with a valid JSON object. Do not include any explanation, formatting, or extra text.
"""
    
    prompt = prompt_template.replace("{context}", context).replace("{question}", question)
    response = llm.invoke(prompt)
    obj = json_to_obj(response.content)
    return obj

def pdf_page_to_image(pdf_path, page_num):
    """Convert a PDF page to an image"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)  # 0-indexed
        
        # Render page to image
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        doc.close()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(img_data))
        return img
    except Exception as e:
        st.error(f"Error rendering PDF page: {str(e)}")
        return None

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG PDF Assistant</h1>
        <p>Upload PDFs, ask questions, and get intelligent answers with source references</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    if st.session_state.model is None or st.session_state.llm is None:
        with st.spinner("Loading AI models..."):
            st.session_state.model, st.session_state.llm = load_models()
    
    # Load index and texts
    if st.session_state.index is None:
        st.session_state.index, st.session_state.texts = load_index()
    
    # Sidebar for file management
    with st.sidebar:
        st.markdown("""
        <div class="card-title">
            📚 Document Management
        </div>
        """, unsafe_allow_html=True)
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file not in st.session_state.uploaded_files:
                    # Save file to docs folder
                    file_path = os.path.join(DOCS_FOLDER, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the PDF
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            chunks = load_and_chunk_pdf(file_path)
                            st.session_state.index, st.session_state.texts = add_text_to_index(
                                chunks, st.session_state.model, st.session_state.index, st.session_state.texts
                            )
                            save_index_and_texts(st.session_state.index, st.session_state.texts)
                            st.session_state.uploaded_files.append(uploaded_file)
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Display uploaded files
        st.markdown("### 📄 Uploaded Files")
        docs_files = list(Path(DOCS_FOLDER).glob("*.pdf"))
        
        if docs_files:
            for file_path in docs_files:
                file_size = file_path.stat().st_size
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div class="file-item">
                        <div class="file-info">
                            <div class="file-name">{file_path.name}</div>
                            <div class="file-size">{format_file_size(file_size)}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"delete_{file_path.name}", help="Delete file"):
                        try:
                            os.remove(file_path)
                            # Rebuild index without this file
                            st.session_state.index = faiss.IndexFlatL2(384)
                            st.session_state.texts = []
                            
                            # Reprocess remaining files
                            remaining_files = list(Path(DOCS_FOLDER).glob("*.pdf"))
                            for remaining_file in remaining_files:
                                chunks = load_and_chunk_pdf(str(remaining_file))
                                st.session_state.index, st.session_state.texts = add_text_to_index(
                                    chunks, st.session_state.model, st.session_state.index, st.session_state.texts
                                )
                            
                            save_index_and_texts(st.session_state.index, st.session_state.texts)
                            st.success(f"Deleted {file_path.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {str(e)}")
        else:
            st.info("No PDF files uploaded yet")
        
        # Clear all files button
        if docs_files and st.button("🗑️ Clear All Files", type="secondary"):
            try:
                for file_path in docs_files:
                    os.remove(file_path)
                
                # Reset index and texts
                st.session_state.index = faiss.IndexFlatL2(384)
                st.session_state.texts = []
                st.session_state.uploaded_files = []
                
                # Remove index files
                if os.path.exists(INDEX_FILE):
                    os.remove(INDEX_FILE)
                if os.path.exists(TEXT_FILE):
                    os.remove(TEXT_FILE)
                
                st.success("All files cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing files: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Question input section
        st.markdown("""
        <div class="question-container">
            <div class="card-title">💬 Ask Questions</div>
            <p style="color: #6b7280; margin-bottom: 1rem;">Ask anything about your uploaded documents</p>
        </div>
        """, unsafe_allow_html=True)
        
        question = st.text_input(
            "Your Question:",
            placeholder="What would you like to know about your documents?",
            label_visibility="collapsed"
        )
        
        ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        
        # Answer display
        if ask_button and question:
            if not st.session_state.texts:
                st.warning("⚠️ Please upload some PDF documents first!")
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer_obj = rag_answer(
                            question, 
                            st.session_state.model, 
                            st.session_state.index, 
                            st.session_state.texts, 
                            st.session_state.llm
                        )
                        
                        # Display answer
                        st.markdown("""
                        <div class="answer-container">
                            <div class="card-title">🤖 Answer</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(answer_obj.get("answer", "No answer generated"))
                        
                        # Display source information
                        if answer_obj.get("filename") and answer_obj.get("page_no"):
                            st.markdown(f"""
                            <div class="source-info">
                                <div class="source-label">📖 Source Information</div>
                                <strong>File:</strong> {answer_obj['filename']}<br>
                                <strong>Page:</strong> {answer_obj['page_no']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Store source info in session state for image display
                            st.session_state.current_source = {
                                'filename': answer_obj['filename'],
                                'page_no': answer_obj['page_no']
                            }
                    
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {str(e)}")
    
    with col2:
        # PDF preview section
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">📄 Source Page Preview</div>
        </div>
        """, unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'current_source') and st.session_state.current_source:
            source_info = st.session_state.current_source
            pdf_path = os.path.join(DOCS_FOLDER, source_info['filename'])
            
            if os.path.exists(pdf_path):
                with st.spinner("Rendering PDF page..."):
                    img = pdf_page_to_image(pdf_path, source_info['page_no'])
                    if img:
                        st.image(img, caption=f"Page {source_info['page_no']} from {source_info['filename']}", use_container_width=True)
                    else:
                        st.error("Failed to render PDF page")
            else:
                st.error("Source PDF file not found")
        else:
            st.info("Ask a question to see the source page preview")
    
    # Statistics
    if st.session_state.texts:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📚 Total Documents", len(set(text['filename'] for text in st.session_state.texts)))
        
        with col2:
            st.metric("📄 Total Pages", len(st.session_state.texts))
        
        with col3:
            st.metric("🔍 Index Size", len(st.session_state.texts))

if __name__ == "__main__":
    main()
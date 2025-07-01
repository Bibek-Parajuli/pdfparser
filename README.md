# 📚 PDF RAG Application – AI-powered PDF Q&A System

This project is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the contents of a PDF document. It parses the PDF, stores text chunks using embeddings in a FAISS index, and uses Google's Gemini (via LangChain) to generate intelligent answers based on the document context.

---

## 🛠 Features

- ✅ Extracts and chunks PDF content
- ✅ Stores and retrieves embeddings using FAISS
- ✅ Uses Google's Gemini model (via `langchain-google-genai`) for responses
- ✅ Simple front-end using raw HTML/CSS
- ✅ Easily switch between Python script (`app.py`) or Jupyter notebook (`test.ipynb`)

---

## 📂 Project Structure

.
├── app.py # Python backend code 
├── main.py # Full web app with raw HTML/CSS
├── test.ipynb # Jupyter Notebook with step-by-step implementation
├── requirements.txt # Python dependencies
├── Unit 4_ Operators and Expression.pdf # Sample PDF file
└── README.md # Project documentation (this file)


---

## 🚀 How to Run the Application

### 1. 🔧 Install dependencies
```bash
pip install -r requirements.txt

2. 🔑 Set up your Gemini API Key
In your Python file (app.py, main.py, or test.ipynb), replace: api_key = Your gemini api key


3. 🧠 Run the Python script

python app.py  for python terminal only 

streamlit run main.py ///for web access

🙋‍♂️ Author
Bibek Parajuli
🔒 Cybersecurity & web Developement student

📜 License
Free for educational and non-commercial use.




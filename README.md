# 🩺 Medical Chatbot

**AI-Powered Medical Assistant using RAG, FAISS, Sentence Transformers, and Mistral**

![Application Screenshot](Img.png)

---

## 📌 Overview

The **Medical RAG Chatbot** is an intelligent AI system designed to provide friendly, medically oriented responses using an advanced Retrieval-Augmented Generation (RAG) pipeline. It combines:

* **FAISS vector search** for fast similarity retrieval
* **SentenceTransformer embeddings**
* **Mistral-7B Instruct** for rewriting and generating answers
* **FastAPI backend**
* **HTML/Jinja2 frontend**

The system workflow:

1. Clean and normalize medical text
2. Expand medical abbreviations
3. Rewrite the user question using Mistral
4. Convert question to embeddings using MiniLM
5. Search similar Q&A using FAISS
6. Build a context-enriched prompt using retrieved data
7. Generate the final medical answer
8. Convert answer into a warm, friendly doctor-like tone

---

# 🧠 System Architecture

```
User
   ↓
Frontend (index.html)
   ↓
FastAPI Backend
   ↓
Mistral → Rewrite Question
   ↓
SentenceTransformer → Embedding
   ↓
FAISS → Retrieve Similar Medical Q&A
   ↓
Prompt Builder
   ↓
Mistral → Raw Answer Generation
   ↓
Mistral → Conversational Doctor Tone
   ↓
Return Final Answer to User
```

---

# 📂 Project Structure

```
DEPI-medical-chatbot-fastapi-frontend/
│
├── main.py                       # FastAPI application and endpoints
├── last_api.py                   # RAG pipeline, embeddings, FAISS, Mistral logic
├── HealthCareMagic-5k.json       # Medical Q&A dataset
├── question_embeddings.npy        # Precomputed embedding vectors
│
├── templates/
│   └── index.html                # Web chat interface
│
├── Img.png                       # Application screenshot
├── UI.png                        # UI design screenshot
├── README.md                     # Documentation file
└── .gitignore
```

---

# 🧬 Dataset

The project uses the **HealthCareMagic** dataset, containing real-world patient questions and doctor responses:

* `input` → Patient question
* `output` → Doctor answer

### Preprocessing includes:

* Lowercasing
* URL removal
* Number removal
* Stopword filtering
* Contraction expansion
* Lemmatization
* Medical abbreviation expansion (e.g., “BP” → “blood pressure”)
* Markdown cleanup

---

# 🧹 Preprocessing Pipeline

### Steps:

1. Remove URLs
2. Clean special characters
3. Remove digits
4. Remove stopwords
5. Normalize medical abbreviations
6. Lemmatize text
7. Clean markdown patterns
8. Expand contractions

This ensures optimized embeddings and better retrieval accuracy.

---

# 📊 Embedding & Vector Indexing

The system uses:

### **Embedding Model:**

`all-MiniLM-L6-v2` — optimized for semantic search

### **FAISS Index:**

`IndexFlatIP` using **Inner Product** similarity
All vectors are **L2-normalized** before indexing.

Process:

1. Encode all dataset questions
2. Normalize embeddings
3. Save to `question_embeddings.npy`
4. Build FAISS index
5. Search for top-k similar entries (`k = 3`)

---

# 🤖 Mistral AI Integration

The chatbot uses three sequential inference steps:

### 1️⃣ Rewrite the user input

* Makes unclear questions medically precise
* Improves retrieval accuracy

### 2️⃣ Generate the final medical answer

* Uses retrieved FAISS context
* Produces accurate medical explanations

### 3️⃣ Apply a conversational doctor tone

* Warm
* Friendly
* Patient-safe wording

**Model:**
`mistralai/Mistral-7B-Instruct-v0.2`

---

# 🧩 Backend (FastAPI)

### Routes:

#### **GET /**

Returns the main chat interface.

#### **POST /api/chat**

Body:

```json
{
  "msg": "user message"
}
```

Response:

```json
{
  "response": "AI doctor's answer"
}
```

### Backend Features:

* Stores last 10 user messages
* Executes full RAG pipeline
* Async FastAPI server
* Integrates HuggingFace InferenceClient
* Clean separation between logic (`last_api.py`) and server (`main.py`)

---

# 💬 Frontend (HTML + Jinja2)

The UI supports:

* Live message sending via Fetch API
* Automatic message display for both user and bot
* Responsive chat layout
* Easy to customize styling

Screenshot:

![UI](UI.png)

---

# 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmadashrafgalal/DEPI-medical-chatbot-fastapi-frontend
cd DEPI-medical-chatbot-fastapi-frontend
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn jinja2
pip install faiss-cpu
pip install sentence-transformers
pip install pandas numpy nltk contractions
pip install huggingface_hub
```

### 3. Run the application

```bash
python main.py
```

Server starts at:

```
http://localhost:8000
```

---

# 🛠 Future Enhancements

* Add user authentication
* Add streaming responses
* Add multilingual support
* Add admin dashboard
* Add chat history persistence
* Add a medical disclaimer block
* Add a model selection panel

---

# 📄 License

MIT License.

---

# 🤝 Contributing

Contributions are welcome!
Feel free to open issues or submit PRs.

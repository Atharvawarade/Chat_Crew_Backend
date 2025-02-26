import os
import json
import requests
import threading
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db
from voice import get_voice_input

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
genai.configure(api_key=api_key)

# Initialize Firebase
firebase_config = {
    "type": os.getenv("FIREBASE_TYPE"),
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('FIREBASE_CLIENT_EMAIL').replace('@', '%40')}"
}

FIREBASE_DB_URL = "https://student-assistance-chatbot-default-rtdb.firebaseio.com/"
cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# Flask app initialization
app = Flask(__name__)
CORS(app, origins=["https://student-assistance-chatbot-chatcrew.netlify.app/"], supports_credentials=True)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret')
Session(app)

# File paths and directories
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_BACKUP_FILE = os.path.join(BACKEND_DIR, "firebase_backup.json")
VECTOR_STORE_PATH = os.path.join(BACKEND_DIR, "faiss_index")
PDF_DIR = os.path.join(BACKEND_DIR, "pdf_files")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)


# MAX_HISTORY_LENGTH = 5  # Maintain last 5 exchanges
# HISTORY_FORMAT = """User: {question}
# Assistant: {answer}"""

# Global variables
vector_store_initialized = False
downloaded_urls = set()
file_lock = threading.Lock()

def initialize_system():
    """Initial setup for first-time execution"""
    print("\n🚀 Starting initial system setup...")
    backup_realtime_database()
    pdf_urls = extract_pdf_urls()
    download_pdfs(pdf_urls)
    create_vector_store()
    print("\n✅ Initial setup completed!")

def backup_realtime_database():
    """Backup Firebase database to JSON file"""
    print("\n📥 Backing up Firebase database...")
    try:
        ref = db.reference("/")
        data = ref.get() or {}
        with open(JSON_BACKUP_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"✅ Database backed up to {JSON_BACKUP_FILE}")
    except Exception as e:
        print(f"❌ Error backing up database: {e}")
        raise

def extract_pdf_urls():
    """Extract PDF URLs from JSON backup"""
    print("\n🔍 Extracting PDF URLs...")
    pdf_urls = []
    
    try:
        with open(JSON_BACKUP_FILE, 'r') as f:
            data = json.load(f)
        
        def recursive_search(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    recursive_search(v)
            elif isinstance(obj, list):
                for item in obj:
                    recursive_search(item)
            elif isinstance(obj, str) and re.match(r"https?://.*\.pdf$", obj):
                pdf_urls.append(obj)

        recursive_search(data)
        print(f"✅ Found {len(pdf_urls)} PDF URLs")
        return pdf_urls
    except Exception as e:
        print(f"❌ Error extracting PDF URLs: {e}")
        return []

def download_pdfs(pdf_urls):
    """Download PDFs with deduplication and improved filename handling"""
    print("\n📥 Downloading PDFs...")
    new_downloads = 0
    
    with file_lock:
        existing_files = set(os.listdir(PDF_DIR))
        existing_urls = {url.split('/')[-1].split('?')[0] for url in downloaded_urls}  # Remove URL params
        
        for url in pdf_urls:
            try:
                # Clean URL and get filename
                clean_url = url.split('?')[0]
                filename = os.path.basename(clean_url)
                
                if not filename.lower().endswith('.pdf'):
                    print(f"⚠️ Skipping non-PDF URL: {url}")
                    continue
                
                if filename in existing_urls or filename in existing_files:
                    continue
                
                response = requests.get(clean_url, stream=True, timeout=30)
                response.raise_for_status()
                
                # Validate PDF content
                if 'application/pdf' not in response.headers.get('Content-Type', ''):
                    print(f"⚠️ Invalid PDF content at {clean_url}")
                    continue
                
                with open(os.path.join(PDF_DIR, filename), 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_urls.add(clean_url)
                new_downloads += 1
                print(f"Downloaded: {filename}")
                
            except Exception as e:
                print(f"❌ Failed to download {url}: {e}")
        
        print(f"✅ Downloaded {new_downloads} new PDFs")


def create_vector_store():
    global vector_store_initialized
    print("\n🧠 Creating vector store...")
    
    try:
        if not os.path.exists(JSON_BACKUP_FILE):
            print("❌ Missing Firebase backup file!")
            raise FileNotFoundError("Firebase backup file not found")

        text = extract_combined_text()
        if len(text) < 100:  # Minimum expected text length
            raise ValueError("Insufficient text data for vectorization")
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        chunks = splitter.split_text(text)
        
        if not chunks:
            print("⚠️ No text chunks to process")
            return

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        with file_lock:  # Add lock for vector store operations
            if vector_store_initialized and os.path.exists(VECTOR_STORE_PATH):
                try:
                    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
                    vector_store.add_texts(chunks)
                except Exception as load_error:
                    print(f"⚠️ Failed to load existing vector store: {load_error}. Creating new one.")
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            else:
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            
            vector_store.save_local(VECTOR_STORE_PATH)
            vector_store_initialized = True
        
        print("✅ Vector store updated!")
    except Exception as e:
        print(f"❌ Critical error in vector store creation: {str(e)}")
        # Add email notification or log to Firebase
        raise

def extract_combined_text():
    """Extract text from JSON backup and PDFs"""
    text = ""
    
    # Process JSON data
    with open(JSON_BACKUP_FILE, 'r') as f:
        data = json.load(f)
        text += json.dumps(data, indent=2) + "\n"
    
    # Process PDFs
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            try:
                reader = PdfReader(os.path.join(PDF_DIR, filename))
                text += f"\nPDF: {filename}\n"
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    
    return text

def firebase_change_handler(event):
    """Handle Firebase realtime updates"""
    print("\n🔄 Firebase change detected! Updating system...")
    try:
        backup_realtime_database()
        pdf_urls = extract_pdf_urls()
        download_pdfs(pdf_urls)
        create_vector_store()
        print("✅ System updated with latest changes!")
    except Exception as e:
        print(f"❌ Error processing Firebase update: {e}")

def start_firebase_listener():
    """Start Firebase realtime listener"""
    print("\n👂 Starting Firebase listener...")
    ref = db.reference("/")
    ref.listen(firebase_change_handler)

# Add these constants at the top with other configurations
MAX_HISTORY_LENGTH = 5  # Maintain last 5 exchanges
HISTORY_FORMAT = """User: {question}
Assistant: {answer}"""

from langchain.chains import LLMChain  # Add this import


def query_vector_store(question):
    try:
        if not os.path.exists(VECTOR_STORE_PATH):
            raise ValueError("Vector store not found. Please initialize it first.")

        print(f"Querying vector store with question: {question}")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(question)
        
        # Build context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        history = session.get("history", "")
        chain = get_conversational_chain()
        
        # Get response from chain
        response = chain.predict(
            history=history,
            context=context,
            question=question
        )

        response_text = response.strip()
        if not response_text:
            return "I couldn't find that information. Could you please rephrase or provide more details?"

        # Update conversation history
        new_entry = HISTORY_FORMAT.format(question=question, answer=response_text)
        if history:
            entries = history.split('\n')
            entries = entries[-(MAX_HISTORY_LENGTH*2-2):]
            entries.append(new_entry)
            session["history"] = '\n'.join(entries)
        else:
            session["history"] = new_entry

        session.modified = True
        return response_text

    except Exception as e:
        print(f"Error processing question: {e}")
        return "I'm having trouble answering that right now. Please try again later."

def get_conversational_chain():
    print("Creating conversational chain...")
    prompt_template = """You are a helpful university assistant. Use the following context and conversation history to answer the question.
If the question refers to previous topics, use the history to understand the context. Keep answers concise and specific.

Conversation History:
{history}

Context:
{context}

Question: {question}

Answer in clear, short sentences:"""
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question"])
    return LLMChain(llm=model, prompt=prompt)


@app.route("/query", methods=["POST"])
def handle_query():
    """API endpoint to handle user queries."""
    try:
        # Check for session initialization
        if 'history' not in session:
            session['history'] = ''

        data = request.json
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"response": "Please ask a question about the university."}), 400

        response_text = query_vector_store(question)
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error in /query endpoint: {e}")
        return jsonify({"response": "There was an error processing your question. Please try again."}), 500

@app.route('/voice-input', methods=['POST'])
def voice_input():
    try:
        recognized_text = get_voice_input()
        if "Error" in recognized_text:
            print(f"Voice input error: {recognized_text}")
            return jsonify({"success": False, "message": recognized_text})
        print(f"Voice input recognized text: {recognized_text}")
        return jsonify({"success": True, "text": recognized_text})
    except Exception as e:
        print(f"Unexpected error in voice input: {e}")
        return jsonify({"success": False, "message": "An unexpected error occurred in voice input processing."})
    

@app.route('/health')
def health_check():
    status = {
        "vector_store": os.path.exists(VECTOR_STORE_PATH),
        "firebase_connected": False,
        "google_api": bool(os.getenv("GOOGLE_API_KEY")),
        "session_secret": bool(app.secret_key)
    }
    
    try:
        db.reference('/').get()
        status['firebase_connected'] = True
    except Exception as e:
        status['firebase_error'] = str(e)
        
    return jsonify(status), 200

@app.route('/check-secret')
def check_secret():
    if not app.secret_key:
        return "Secret key not configured!", 500
    return "Secret key is properly configured!", 200    


@app.route('/debug/vector-store')
def debug_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return jsonify({
            "status": "exists",
            "documents": len(vector_store.docstore._dict),
            "last_modified": os.path.getmtime(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/debug/rebuild-vector-store')
def rebuild_vector_store():
    try:
        create_vector_store()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500  


if __name__ == "__main__":
    # Initial setup
    initialize_system()
    
    # Start Firebase listener in background
    firebase_thread = threading.Thread(target=start_firebase_listener, daemon=True)
    firebase_thread.start()
    
    # Start Flask app
    print("\n🚀 Server is ready to handle requests!")
    app.run(port=5000)
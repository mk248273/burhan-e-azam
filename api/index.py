from flask import Flask, request, jsonify, send_from_directory,render_template
from flask_cors import CORS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import random
import google.generativeai as genai

# Load environment variables
load_dotenv()

GROQ_API_KEY = ""

# Initialize Flask app
app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)
CORS(app)



@app.route('/')
def index():
    return render_template('index.html')

# Modify your app initialization to include the template folder

# Constants
CHURCH_NAME = "WholeLife Church"
ASSISTANT_NAME = "Gaby"

class ChurchAssistant:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.chain = None
        self.prompt_template = None
        self.initialize_components()

    def initialize_components(self):
        # Create the prompt template first
        template = f"""
        System Instructions for {CHURCH_NAME} Assistant:

        1. Role: You are {ASSISTANT_NAME}, the official virtual assistant for {CHURCH_NAME}.
        2. Response Rule: Answer ONLY using information from the provided context.
        3. Knowledge Limit: If information isn't in context, say "I apologize, but I don't have that specific information. Please contact our church office for assistance."
        4. Tone: Maintain warm, professional, and empathetic communication.
        5. Quality: Ensure responses are grammatically correct and professionally written.
        6. Privacy: Protect sensitive information and maintain confidentiality.
        7. Values: Every response should reflect church values of faith, community, and service.

        Context: {{context}}
        Question: {{question}}

        Answer:
        """

        self.prompt_template = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Initialize the language model
        model = ChatGroq(
        api_key="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",
            model_name="llama-3.3-70b-versatile",
        )

        # Create the QA chain
        self.chain = load_qa_chain(
            llm=model,
            chain_type="stuff",
            prompt=self.prompt_template
        )

        # Try to load existing vector store
        try:
            self.vector_store = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except:
            print("No existing vector store found. Please add documents using /update_knowledge endpoint.")

    def process_pdf(self, pdf_path):
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        
        self.vector_store = FAISS.from_texts(
            chunks,
            embedding=self.embeddings
        )
        self.vector_store.save_local("faiss_index")

    def get_response(self, question):
        if not self.vector_store:
            return {
                "output_text": "I apologize, but my knowledge base hasn't been initialized yet. Please contact the administrator."
            }

        docs = self.vector_store.similarity_search(question, k=5)
        response = self.chain(
            {
                "input_documents": docs,
                "question": question
            },
            return_only_outputs=True
        )
        return response

# Greeting messages
GREETINGS = [
    f"Hi! I'm {ASSISTANT_NAME}, your friendly assistant at {CHURCH_NAME}. How can I help you today?",
    f"Welcome to {CHURCH_NAME}! I'm {ASSISTANT_NAME}, and I'm here to assist you.",
    f"Hello! I'm {ASSISTANT_NAME}, your guide to everything at {CHURCH_NAME}. What can I help you with?",
    f"Greetings! This is {ASSISTANT_NAME} from {CHURCH_NAME}. How may I serve you today?"
]

# Quick replies data
QUICK_REPLIES = {
    "prayer_pastoral": {
        "text": "I need prayer or pastoral care",
        "response": {
            "contact": """pastoral.care@wholelife.church, care@wholelife.church, prayer@wholelife.church"""
        }
    },
    "membership": {
        "text": "Join or transfer membership",
        "response": {
            "contact": "https://share.fluro.io/form/6762d242e959460036fc930b"
        }
    },
    "small_groups": {
        "text": "Connect with Small Community",
        "response": {
            "contact": "https://wholelife.church/communities"
        }
    },
    "volunteer": {
        "text": "Volunteer opportunities",
        "response": {
            "contact": "https://wholelife.church/serve",
        }
    },
    "bible_baptism": {
        "text": "Bible study & baptism",
        "response": {
            " ": "ken@wholelife.church",
        }
    },
    "child_dedication": {
        "text": "Child dedication",
        "response": {
            "contact": "melanie@wholelife.church",
        }
    },
    "events": {
        "text": "Upcoming events",
        "response": {
            "contact": "https://wholelife.church/events",
        }
    }
}


# Initialize the assistant
assistant = ChurchAssistant()

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/start', methods=['GET'])
def start_chat():
    greeting = random.choice(GREETINGS)
    return jsonify({
        "status": "success",
        "start chat": greeting
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0"
    })

@app.route('/df_ask', methods=['POST'])
def dialogflow_ask():
    try:
        data = request.get_json()
        question = data.get('question', '').lower()

        for key, reply_data in QUICK_REPLIES.items():
            if question.strip() in reply_data['text'].lower():
                return jsonify({
                    "status": "success",
                    "response": reply_data['response']
                })

        return jsonify({
            "status": "error",
            "response": "I'm not sure about that. Could you please rephrase your question?"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('query')

        if not question:
            return jsonify({
                "status": "error",
                "error": "No question provided"
            }), 400

        response = assistant.get_response(question)
        
        return jsonify({
            "status": "success",
            "response": response
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/update_knowledge', methods=['POST'])
def update_knowledge():
    try:
        if 'file' not in request.files:
            return jsonify({
                "status": "error",
                "error": "No file provided"
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "status": "error",
                "error": "No file selected"
            }), 400

        if not file.filename.endswith('.pdf'):
            return jsonify({
                "status": "error",
                "error": "Only PDF files are supported"
            }), 400

        temp_path = "temp.pdf"
        file.save(temp_path)
        assistant.process_pdf(temp_path)
        os.remove(temp_path)

        return jsonify({
            "status": "success",
            "message": "Knowledge base updated successfully"
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    os.makedirs("static/images", exist_ok=True)
    app.run(
        host='0.0.0.0',
        port=7888,
    
    )
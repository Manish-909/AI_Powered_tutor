from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import os
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class StreamCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
    
    def generate_response(self):
        for token in self.tokens:
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

def get_vector_store():
    # Create directories if they don't exist
    os.makedirs("course_data", exist_ok=True)
    os.makedirs("vector_cache", exist_ok=True)
    
    pdf_path = "course_data/ai_course.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at {pdf_path}")
        return None

    try:
        with open(pdf_path, "rb") as f:
            pdf_hash = hashlib.md5(f.read()).hexdigest()
        cache_path = f"vector_cache/{pdf_hash}.faiss"

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        if os.path.exists(cache_path):
            logger.info("Loading cached vector store")
            return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("Creating new vector store")
            loader = PyPDFLoader(pdf_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = loader.load_and_split(splitter)
            vector_store = FAISS.from_documents(docs, embeddings)
            vector_store.save_local(cache_path)
            return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

# Initialize vector store at startup
vector_store = get_vector_store()

@app.route('/ask', methods=['GET'])
def ask_question():
    if not vector_store:
        return Response("data: " + json.dumps({"token": "System error: Failed to load course content. Make sure ai_course.pdf exists in course_data folder."}) + "\n\n", 
                      mimetype="text/event-stream")

    question = request.args.get('question')
    if not question:
        return Response("data: " + json.dumps({"token": "Please enter a question"}) + "\n\n",
                      mimetype="text/event-stream")

    try:
        callback = StreamCallback()
        
        # Initialize LLM with error handling
        try:
            llm = Ollama(
                model="mistral",
                base_url="http://localhost:11434",
                temperature=0.3,
                callbacks=[callback],
                timeout=120  # Increased timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            return Response("data: " + json.dumps({"token": "AI service unavailable. Please ensure Ollama is running."}) + "\n\n",
                          mimetype="text/event-stream")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            callbacks=[callback]
        )

        # Process the question
        qa_chain.invoke({
            "query": f"""
            You are an AI tutor for an Introduction to Artificial Intelligence course.
            Strictly answer questions based only on the course material.
            If the question is unrelated to the course content, respond:
            "I can only answer questions about the AI course material."
            
            Question: {question}
            """
        })

        return Response(callback.generate_response(), mimetype="text/event-stream")

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return Response("data: " + json.dumps({"token": f"Error processing your question: {str(e)}"}) + "\n\n",
                      mimetype="text/event-stream")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "ollama_available": True,
        "vector_store_loaded": vector_store is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
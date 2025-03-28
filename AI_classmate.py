from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import json
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)

OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
MODEL_NAME = "mistral"  # Model to use

@app.route("/ask", methods=["GET", "POST"])
def ask_ai():
    if request.method == "GET":
        user_question = request.args.get("question")  # Get question from query parameters
    else:
        data = request.get_json()
        user_question = data.get("question")  # Get question from JSON body

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    payload = {"model": MODEL_NAME, "prompt": user_question, "stream": True}  # Enable streaming

    try:
        # Log the request
        app.logger.debug(f"Sending request to Ollama: {payload}")

        # Stream the response from Ollama
        response = requests.post(OLLAMA_API_URL, json=payload, stream=True)

        def generate():
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield f"data: {json.dumps({'reply': data['response']})}\n\n"  # Stream each part
                            full_response += data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            yield f"data: {json.dumps({'done': True})}\n\n"  # Signal end of stream

        return Response(generate(), mimetype="text/event-stream")  # Use Server-Sent Events (SSE)

    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app, origins="https://mindforce.decipherinc.com", supports_credentials=True)

# Initialize LLM
llm = Ollama(model="llama3")

# Prompt template
template = PromptTemplate(
    input_variables=["questions"],
    template="Please summarize the following survey questions in 3-4 lines:\n\n{questions}"
)

# LangChain chain
chain = template | llm

@app.after_request
def add_cors_headers(response):
    # This ensures headers are returned even if Flask-CORS fails
    response.headers["Access-Control-Allow-Origin"] = "https://mindforce.decipherinc.com"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/summarize', methods=['POST', 'OPTIONS'])
def summarize():
    if request.method == 'OPTIONS':
        # Respond to preflight CORS request
        return '', 204

    data = request.get_json()
    questions = data.get("questions", [])

    if not questions:
        return jsonify({"summary": "No questions provided."}), 400

    try:
        joined = "\n".join(questions)
        summary = chain.invoke({"questions": joined})
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"summary": f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

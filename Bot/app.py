from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Initialize OpenAI LLM using LangChain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define prompt template
template = PromptTemplate(
    input_variables=["questions"],
    template="Please summarize the following survey questions in 3-4 lines:\n\n{questions}"
)

# Chain: prompt -> model
chain: Runnable = template | llm

# Summary endpoint
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    questions = data.get("questions", [])

    if not questions:
        return jsonify({"summary": "No questions provided."}), 400

    try:
        joined_questions = "\n".join(questions)
        summary = chain.invoke({"questions": joined_questions})
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"summary": f"Error: {str(e)}"}), 500

# Health check (optional)
@app.route('/')
def home():
    return "Survey Summary Bot is live!"

# Run locally (optional)
if __name__ == '__main__':
    app.run(debug=True)

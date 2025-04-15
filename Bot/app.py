from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os

app = Flask(__name__)
CORS(app)

# Initialize OpenAI model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# Prompt template
template = PromptTemplate(
    input_variables=["questions"],
    template="Please summarize the following survey questions in 3-4 lines:\n\n{questions}"
)

# Chain
chain = LLMChain(llm=llm, prompt=template)

@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize():
    if request.method == 'OPTIONS':
        return '', 204

    data = request.json
    questions = data.get("questions", [])

    if not questions:
        return jsonify({"summary": "No questions provided."}), 400

    try:
        joined_questions = "\n".join(questions)
        summary = chain.run(questions=joined_questions)
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"summary": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

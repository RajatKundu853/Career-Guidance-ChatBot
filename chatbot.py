from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt4all import GPT4All  # Import GPT4All

app = Flask(__name__)
CORS(app)  # Allow frontend access

# Load GPT4All model
model_path = r"D:\ChatBot\mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Update with your model filename
# Force CPU mode (disable CUDA)
gpt4all_model = GPT4All(model_path, device="cpu", allow_download=False)

print("Model loaded successfully in CPU mode!")

def chatbot_response(user_input):
    """ Generate chatbot response using GPT4All """
    response = gpt4all_model.generate(user_input, max_tokens=200)
    return response

@app.route('/chatbot', methods=['POST'])
def chat():
    """ API endpoint to handle chatbot requests """
    data = request.get_json()
    user_input = data.get("query", "")

    if not user_input:
        return jsonify({"error": "No query provided"}), 400

    response = chatbot_response(user_input)
    return jsonify({"answer": response.strip()})

if __name__ == '__main__':
    app.run(debug=True)

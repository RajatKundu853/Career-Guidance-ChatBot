import threading
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt4all import GPT4All  # Import GPT4All

app = Flask(__name__)
CORS(app)  # Allow frontend access

# Load GPT4All model
model_path = r"model_path or location\\model_name.gguf"  # Update with your model filename & path
# Force CPU mode (disable CUDA)
gpt4all_model = GPT4All(model_path, device="cpu", allow_download=False)

print("Model loaded successfully in CPU mode!")


def show_spinner(stop_event):
    """ Show spinner or progress dots while waiting """
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_event.is_set():
        print(f"\r[DEBUG] Generating response... {spinner[idx % len(spinner)]}", end='', flush=True)
        idx += 1
        time.sleep(0.1)
    print("\r[DEBUG] Response generated.          ")

def chatbot_response(user_input):
    """ Generate chatbot response using GPT4All with spinner """
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=show_spinner, args=(stop_event,))
    spinner_thread.start()

    try:
        response = gpt4all_model.generate(
            user_input,
            max_tokens=200,
            temp=0.7,
            top_k=40,
            top_p=0.9,
            streaming=False
        )
    finally:
        stop_event.set()
        spinner_thread.join()

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


print("Run index.html")


if __name__ == '__main__':
    app.run(debug=True)

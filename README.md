# Career-Guidance-ChatBot
An AI based chatbot which help about career related problem.

## Prerequisites  
- Windows OS  
- Python 3.8 or later  

## Installation Steps  

### 1. Open Command Prompt  
Press `Win + R`, type `cmd`, and hit **Enter**.  

### 2. Check if Python is Installed  
Run the following command:  
```sh
python --version
```
If Python is not installed, download and install it from https://www.python.org/

### 3. Install Required Packages
Run the following command to install Flask, Flask-CORS, GPT4All, and PyTorch libraries:
```sh
pip install flask flask-cors gpt4all torch torchvision torchaudio
```
### 4. Verify GPT4All Installation
Check if gpt4all is installed by running:
```sh
pip show gpt4all
```
### 5. Install GPT4All GUI
Download and install gpt4all-installer-win64.exe from the official GPT4All website: https://www.nomic.ai/gpt4all

### 6. Restart the Computer
After installation, restart your computer to apply changes.

### 7. Download & Configure the Model 
Download the model: ```mistral-7b-instruct-v0.2.Q4_K_M.gguf``` or  ```mistral-7b-instruct-v0.1.Q4_0.gguf``` or  ```tinyllama-1.1b-chat-v1.0.Q4_0.gguf```   from   https://huggingface.co/   and Locate your model file and update the model path in ```chatbot.py``` with correct path like : ```model_path = "C:/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf" ```
Save the file after making changes.
#### [ you can use another model depend on your purpose ] 

### 8. Run the Chatbot
Once setup is complete, you can start your chatbot by running ```chatbot.py```

### 9. Open in GUI mode
Open ```index.html``` with any browser & use it.

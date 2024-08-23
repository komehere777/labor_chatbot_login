from flask import Flask, render_template, request, jsonify, session
from utils import *
from config import SECRET_KEY

app = Flask(__name__)

app.secret_key = SECRET_KEY

@app.route('/')
def index():
    # 벡터 저장소 초기화 시에만 호출
    initialize_vectorstore()

    return render_template('index.html')

@app.route('/chat')
def chat():
    
    session.clear()

    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    # 유저가 보낸 메시지를 받음
    user_input = request.json.get("message")

    chat_history = session.get('chat_history', '')
    
    ai_response, chat_history = get_ai_response(user_input, chat_history)
    ai_response = ai_response.replace("\n", "<br>")

    session['chat_history'] = chat_history
    
    # JSON 형태로 응답을 반환
    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)

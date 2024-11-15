from flask import Flask, render_template, request, jsonify
from query_data import query_rag
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

app.config['JSON_AS_ASCII'] = False
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({'response': 'Nhập câu hỏi...'})
    
    try:
        bot_response = query_rag(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
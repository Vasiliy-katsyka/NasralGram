import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sqlalchemy import or_, and_, desc
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

# --- SETUP ---

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='', template_folder='.')
CORS(app, resources={r"/*": {"origins": "https://vasiliy-katsyka.github.io"}})

# Configure Database from environment variable
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Configure Google Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# --- DATABASE MODELS ---

# Association table for many-to-many relationship between Users and Chats
chat_participants = db.Table('chat_participants',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('chat_id', db.Integer, db.ForeignKey('chat.id'), primary_key=True)
)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True) # Indexed for faster search
    chats = db.relationship('Chat', secondary=chat_participants, back_populates='participants')

    def to_dict(self):
        return {"id": self.id, "username": self.username}

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    participants = db.relationship('User', secondary=chat_participants, back_populates='chats')
    messages = db.relationship('Message', backref='chat', lazy='dynamic')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True) # Indexed for sorting
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False, index=True) # Indexed for fast message retrieval
    sender = db.relationship('User')

    def to_dict(self):
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "chat_id": self.chat_id,
            "sender_username": self.sender.username
        }
        
# --- API ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({"error": "Username is required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409
        
    new_user = User(username=username)
    db.session.add(new_user)
    db.session.commit()
    return jsonify(new_user.to_dict()), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())

@app.route('/users/search', methods=['GET'])
def search_users():
    query = request.args.get('q', '')
    current_user_id = request.args.get('current_user_id')
    if not query:
        return jsonify([])
        
    # Search for users but exclude the current user from results
    users = User.query.filter(User.username.ilike(f'%{query}%'), User.id != current_user_id).limit(10).all()
    return jsonify([user.to_dict() for user in users])

@app.route('/chats', methods=['GET'])
def get_chats():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    # This is an efficient query to get chat partners and avoids the N+1 problem
    chats_data = []
    for chat in user.chats:
        # Find the other participant in the chat
        other_participant = next((p for p in chat.participants if p.id != user.id), None)
        if other_participant:
            chat_name = other_participant.username
            chat_id = chat.id
            chats_data.append({"id": chat_id, "name": chat_name})
            
    return jsonify(chats_data)

@app.route('/chats/start', methods=['POST'])
def start_chat():
    data = request.json
    user1_id = data.get('user1_id')
    user2_id = data.get('user2_id')

    user1 = User.query.get(user1_id)
    user2 = User.query.get(user2_id)

    if not user1 or not user2:
        return jsonify({"error": "One or more users not found"}), 404

    # Find if a chat already exists between these two users
    # This subquery finds chats that user1 is in.
    # The main query then checks if any of those chats also contain user2.
    existing_chat = Chat.query.join(chat_participants).filter(
        and_(
            Chat.id == chat_participants.c.chat_id,
            chat_participants.c.user_id == user1_id
        )
    ).join(chat_participants, Chat.id == chat_participants.c.chat_id).filter(
        chat_participants.c.user_id == user2_id
    ).first()


    if existing_chat:
        return jsonify({"chat_id": existing_chat.id, "name": user2.username})

    # If no chat exists, create a new one
    new_chat = Chat()
    new_chat.participants.append(user1)
    new_chat.participants.append(user2)
    db.session.add(new_chat)
    db.session.commit()
    
    return jsonify({"chat_id": new_chat.id, "name": user2.username}), 201

@app.route('/chats/<int:chat_id>/messages', methods=['GET'])
def get_messages(chat_id):
    # Fetch messages, ordered by timestamp. Using indexes on chat_id and timestamp makes this fast.
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.timestamp.asc()).all()
    return jsonify([message.to_dict() for message in messages])

@app.route('/messages', methods=['POST'])
def post_message():
    data = request.json
    new_message = Message(
        content=data['content'],
        sender_id=data['sender_id'],
        chat_id=data['chat_id']
    )
    db.session.add(new_message)
    db.session.commit()
    return jsonify(new_message.to_dict()), 201

@app.route('/gemini-chat', methods=['POST'])
def gemini_chat():
    data = request.json
    user_message = data.get('content')
    
    if not user_message:
        return jsonify({"error": "No content provided"}), 400

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(user_message)
        
        return jsonify({
            "sender_username": "Gemini",
            "content": response.text
        })
    except Exception as e:
        app.logger.error(f"Gemini API error: {e}")
        return jsonify({"error": "Failed to get response from Gemini"}), 500

# To reduce compute on Neon.tech, it automatically scales to zero when inactive.
# The main cost is "Compute Time". We reduce it by:
# 1. Using indexed columns (username, timestamp, chat_id) for faster queries.
# 2. Using efficient queries (like the one in /chats) to avoid N+1 problems.
# 3. Connection pooling is handled by SQLAlchemy, reusing connections efficiently.
# 4. For a production app, you might add caching (e.g., with Redis) for frequently accessed, non-changing data.

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)

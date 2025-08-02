import os
import psycopg2
import psycopg2.extras
import jwt
import base64
from datetime import datetime, timedelta, timezone
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, g, send_from_directory
from dotenv import load_dotenv
import google.generativeai as genai

# --- SETUP ---
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

# Configure Gemini AI Client
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    gemini_model = None

# --- DATABASE ---
def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(DATABASE_URL)
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    # Corrected line: 'is not' is the correct operator for checking identity against None.
    if db is not None:
        db.close()

def setup_database():
    db = psycopg2.connect(DATABASE_URL)
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            bio TEXT,
            profile_picture BYTEA
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id SERIAL PRIMARY KEY,
            is_group BOOLEAN DEFAULT FALSE,
            group_name TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_participants (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
            UNIQUE(user_id, chat_id)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
            sender_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            content_type TEXT NOT NULL, -- 'text', 'image', 'file', 'ai_response'
            text_content TEXT,
            media_content BYTEA,
            media_filename TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    db.commit()
    cur.close()
    db.close()
    print("Database tables checked/created.")

# --- MIDDLEWARE & AUTH ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            db = get_db()
            cur = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute("SELECT * FROM users WHERE id = %s", (data['user_id'],))
            g.current_user = cur.fetchone()
            cur.close()
            if g.current_user is None:
                return jsonify({'message': 'User not found!'}), 401
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- ROUTES ---
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Username and password are required'}), 400

    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (data['username'], hashed_password)
        )
        db.commit()
    except psycopg2.IntegrityError:
        return jsonify({'message': 'Username already exists'}), 409
    finally:
        cur.close()

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Could not verify'}), 401
    
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM users WHERE username = %s", (data['username'],))
    user = cur.fetchone()
    cur.close()

    if not user or not check_password_hash(user['password_hash'], data['password']):
        return jsonify({'message': 'Invalid username or password'}), 401

    token = jwt.encode({
        'user_id': user['id'],
        'exp': datetime.now(timezone.utc) + timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm="HS256")

    return jsonify({'token': token, 'username': user['username']})

@app.route('/api/profile', methods=['GET'])
@token_required
def get_my_profile():
    user = g.current_user
    profile_picture_b64 = base64.b64encode(user['profile_picture']).decode('utf-8') if user['profile_picture'] else None
    return jsonify({
        'username': user['username'],
        'bio': user['bio'],
        'profile_picture': profile_picture_b64
    })

@app.route('/api/profile/<username>', methods=['GET'])
@token_required
def get_user_profile(username):
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT username, bio, profile_picture FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    profile_picture_b64 = base64.b64encode(user['profile_picture']).decode('utf-8') if user['profile_picture'] else None
    return jsonify({
        'username': user['username'],
        'bio': user['bio'],
        'profile_picture': profile_picture_b64
    })

@app.route('/api/profile', methods=['PUT'])
@token_required
def update_profile():
    data = request.get_json()
    bio = data.get('bio')
    profile_picture_b64 = data.get('profile_picture') # Expect base64 string
    
    picture_data = None
    if profile_picture_b64:
        # The base64 string might contain a data URL header, remove it
        if ',' in profile_picture_b64:
            profile_picture_b64 = profile_picture_b64.split(',')[1]
        try:
            picture_data = base64.b64decode(profile_picture_b64)
        except Exception as e:
            return jsonify({'message': 'Invalid base64 for profile picture', 'error': str(e)}), 400

    db = get_db()
    cur = db.cursor()
    if bio is not None and picture_data:
        cur.execute("UPDATE users SET bio = %s, profile_picture = %s WHERE id = %s", (bio, picture_data, g.current_user['id']))
    elif bio is not None:
        cur.execute("UPDATE users SET bio = %s WHERE id = %s", (bio, g.current_user['id']))
    elif picture_data:
        cur.execute("UPDATE users SET profile_picture = %s WHERE id = %s", (picture_data, g.current_user['id']))
    
    db.commit()
    cur.close()
    return jsonify({'message': 'Profile updated successfully'})

@app.route('/api/chats', methods=['GET'])
@token_required
def get_chats():
    user_id = g.current_user['id']
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Fetch all chats the user is a part of
    cur.execute("""
        SELECT c.id, c.is_group, c.group_name
        FROM chats c
        JOIN chat_participants cp ON c.id = cp.chat_id
        WHERE cp.user_id = %s
    """, (user_id,))
    chats = cur.fetchall()
    
    chat_list = []
    for chat in chats:
        chat_info = {'id': chat['id'], 'is_group': chat['is_group']}
        if chat['is_group']:
            chat_info['name'] = chat['group_name']
            # Optionally, fetch participant count or other group info
        else:
            # For 1-on-1 chats, find the other participant's name
            cur.execute("""
                SELECT u.username, u.profile_picture 
                FROM users u
                JOIN chat_participants cp ON u.id = cp.user_id
                WHERE cp.chat_id = %s AND cp.user_id != %s
            """, (chat['id'], user_id))
            other_user = cur.fetchone()
            chat_info['name'] = other_user['username'] if other_user else 'Unknown User'
            chat_info['profile_picture'] = base64.b64encode(other_user['profile_picture']).decode('utf-8') if other_user and other_user['profile_picture'] else None
        
        # Get the last message for preview
        cur.execute("""
            SELECT text_content, content_type, timestamp FROM messages 
            WHERE chat_id = %s ORDER BY timestamp DESC LIMIT 1
        """, (chat['id'],))
        last_message = cur.fetchone()
        chat_info['last_message'] = last_message['text_content'] if last_message and last_message['text_content'] else f"<{last_message['content_type']}>" if last_message else "No messages yet."
        chat_info['last_message_time'] = last_message['timestamp'].isoformat() if last_message else None

        chat_list.append(chat_info)
        
    cur.close()
    return jsonify(chat_list)


@app.route('/api/chats/create', methods=['POST'])
@token_required
def create_chat():
    data = request.get_json()
    # usernames should be a list of usernames to include in the chat
    usernames = data.get('usernames', [])
    group_name = data.get('group_name')

    if not usernames:
        return jsonify({'message': 'Usernames are required'}), 400

    all_users = [g.current_user['username']] + [u for u in usernames if u != g.current_user['username']]
    
    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Get user IDs for all usernames
    cur.execute("SELECT id, username FROM users WHERE username = ANY(%s)", (all_users,))
    participants = cur.fetchall()
    if len(participants) != len(all_users):
        found_users = {p['username'] for p in participants}
        missing_users = [u for u in all_users if u not in found_users]
        return jsonify({'message': f'Users not found: {", ".join(missing_users)}'}), 404
    
    participant_ids = [p['id'] for p in participants]
    is_group = len(participant_ids) > 2

    # For 1-on-1, check if chat already exists
    if not is_group:
        cur.execute("""
            SELECT chat_id FROM chat_participants cp1
            JOIN chat_participants cp2 ON cp1.chat_id = cp2.chat_id
            WHERE cp1.user_id = %s AND cp2.user_id = %s AND (
                SELECT COUNT(*) FROM chat_participants WHERE chat_id = cp1.chat_id
            ) = 2
        """, (g.current_user['id'], participant_ids[1]))
        existing_chat = cur.fetchone()
        if existing_chat:
            return jsonify({'message': 'Chat already exists', 'chat_id': existing_chat['chat_id']}), 200

    # Create the new chat
    cur.execute(
        "INSERT INTO chats (is_group, group_name) VALUES (%s, %s) RETURNING id",
        (is_group, group_name if is_group else None)
    )
    chat_id = cur.fetchone()['id']
    
    # Add participants
    args_str = ','.join(cur.mogrify("(%s,%s)", (user_id, chat_id)).decode('utf-8') for user_id in participant_ids)
    cur.execute("INSERT INTO chat_participants (user_id, chat_id) VALUES " + args_str)
    
    db.commit()
    cur.close()
    
    return jsonify({'message': 'Chat created', 'chat_id': chat_id}), 201


@app.route('/api/messages/<int:chat_id>', methods=['GET'])
@token_required
def get_messages(chat_id):
    # Polling for new messages
    last_id = request.args.get('last_id', 0, type=int)

    db = get_db()
    cur = db.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute("""
        SELECT m.id, m.sender_id, u.username, m.content_type, m.text_content, m.media_content, m.media_filename, m.timestamp 
        FROM messages m
        JOIN users u ON m.sender_id = u.id
        WHERE m.chat_id = %s AND m.id > %s
        ORDER BY m.timestamp ASC
    """, (chat_id, last_id))
    messages = cur.fetchall()
    cur.close()

    result = []
    for msg in messages:
        message_data = {
            'id': msg['id'],
            'sender_username': msg['username'],
            'content_type': msg['content_type'],
            'text_content': msg['text_content'],
            'timestamp': msg['timestamp'].isoformat()
        }
        if msg['media_content']:
            media_b64 = base64.b64encode(msg['media_content']).decode('utf-8')
            # Determine content type for data URL
            ct = msg['content_type'].lower()
            mime_type = 'application/octet-stream'
            if ct == 'image': mime_type = 'image/png' # A simplification
            
            message_data['media_data_url'] = f"data:{mime_type};base64,{media_b64}"
            message_data['media_filename'] = msg['media_filename']

        result.append(message_data)
        
    return jsonify(result)

@app.route('/api/messages', methods=['POST'])
@token_required
def post_messages():
    # This endpoint receives a BATCH of messages
    data = request.get_json()
    messages_to_send = data.get('messages', [])
    chat_id = data.get('chat_id')
    if not messages_to_send or not chat_id:
        return jsonify({'message': 'Invalid batch message request'}), 400

    db = get_db()
    cur = db.cursor()
    
    for msg in messages_to_send:
        content_type = msg.get('content_type')
        text_content = msg.get('text_content')
        media_b64 = msg.get('media_content') # Expect base64
        media_filename = msg.get('media_filename')
        
        media_data = None
        if media_b64:
            try:
                media_data = base64.b64decode(media_b64)
            except:
                continue # Skip malformed messages
        
        cur.execute(
            """INSERT INTO messages (chat_id, sender_id, content_type, text_content, media_content, media_filename)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (chat_id, g.current_user['id'], content_type, text_content, media_data, media_filename)
        )
        
    db.commit()
    cur.close()
    return jsonify({'message': f'{len(messages_to_send)} messages saved'}), 201

@app.route('/api/ai', methods=['POST'])
@token_required
def ask_ai():
    if not gemini_model:
        return jsonify({'error': 'AI model is not configured on the server'}), 503
        
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
        
    try:
        response = gemini_model.generate_content(prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': f'Failed to generate content: {str(e)}'}), 500


if __name__ == '__main__':
    setup_database()
    app.run(debug=True, port=5001)

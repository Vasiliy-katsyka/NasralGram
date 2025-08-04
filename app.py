import os
import datetime
import io
import logging
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, request, jsonify, render_template, session,
                   send_file, abort)
from flask_socketio import SocketIO, emit, join_room, leave_room
from psycopg2 import pool
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

# --- Initialization & Configuration ---

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_super_secret_key_for_dev')
DATABASE_URL = os.environ.get('DATABASE_URL')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Configure Gemini AI
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    logging.warning("GEMINI_API_KEY not found. AI features will be disabled.")

# Use eventlet for async mode, as required by gunicorn worker
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Database Setup ---

db_pool = None

def init_db_pool():
    global db_pool
    db_pool = pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)

def get_db_connection():
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()

def release_db_connection(conn):
    db_pool.putconn(conn)

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(256) NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS participants (
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    PRIMARY KEY (user_id, chat_id)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    sender_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    content TEXT,
                    message_type VARCHAR(10) NOT NULL DEFAULT 'text',
                    file_data BYTEA,
                    file_name VARCHAR(255),
                    file_mime_type VARCHAR(100),
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (chat_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender_id ON messages (sender_id);")

            conn.commit()
            logging.info("Database tables checked and created if not exist.")
    finally:
        release_db_connection(conn)

# --- Decorators for Authentication ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return abort(401, description="User not logged in.")
        return f(*args, **kwargs)
    return decorated_function

# --- Helper Functions ---

def is_user_in_chat(user_id, chat_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM participants WHERE user_id = %s AND chat_id = %s",
                (user_id, chat_id)
            )
            return cur.fetchone() is not None
    finally:
        release_db_connection(conn)

def _broadcast_new_message(message_data):
    """Helper to broadcast a message to the correct room."""
    socketio.emit('new_message', message_data, room=str(message_data['chat_id']))

# --- Flask Routes (API) ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required."}), 400

    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, hashed_password)
            )
            conn.commit()
        return jsonify({"success": True}), 201
    except Exception as e:
        conn.rollback()
        logging.error(f"Registration error: {e}")
        return jsonify({"success": False, "error": "Username already exists."}), 409
    finally:
        release_db_connection(conn)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            if user and check_password_hash(user[1], password):
                session['user_id'] = user[0]
                session['username'] = username
                return jsonify({"success": True, "username": username})
            else:
                return jsonify({"success": False, "error": "Invalid credentials"}), 401
    finally:
        release_db_connection(conn)

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/check_session')
def check_session():
    if 'user_id' in session:
        return jsonify({"loggedIn": True, "username": session['username']})
    return jsonify({"loggedIn": False})

@app.route('/get_user_chats')
@login_required
def get_user_chats():
    user_id = session['user_id']
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Query to get chats, other participant's name, and the last message
            cur.execute("""
                WITH last_messages AS (
                    SELECT
                        chat_id,
                        content,
                        message_type,
                        timestamp,
                        ROW_NUMBER() OVER(PARTITION BY chat_id ORDER BY timestamp DESC) as rn
                    FROM messages
                )
                SELECT
                    c.id as chat_id,
                    u.username as participant_username,
                    CASE 
                        WHEN lm.message_type = 'file' THEN 'File'
                        ELSE lm.content
                    END as last_message_content,
                    lm.timestamp as last_message_timestamp
                FROM participants p_me
                JOIN participants p_other ON p_me.chat_id = p_other.chat_id AND p_me.user_id != p_other.user_id
                JOIN users u ON p_other.user_id = u.id
                JOIN chats c ON p_me.chat_id = c.id
                LEFT JOIN last_messages lm ON c.id = lm.chat_id AND lm.rn = 1
                WHERE p_me.user_id = %s
                ORDER BY COALESCE(lm.timestamp, c.created_at) DESC;
            """, (user_id,))
            chats = cur.fetchall()
            chat_list = [
                {"chat_id": row[0], "participant_username": row[1], "last_message": row[2] or "", "timestamp": row[3]}
                for row in chats
            ]
            return jsonify(chat_list)
    finally:
        release_db_connection(conn)

@app.route('/get_chat_messages/<int:chat_id>')
@login_required
def get_chat_messages(chat_id):
    user_id = session['user_id']
    if not is_user_in_chat(user_id, chat_id):
        return abort(403)

    before_id = request.args.get('before', type=int)
    limit = 50

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT m.id, m.content, m.message_type, m.file_name, m.timestamp, u.username as sender_username
                FROM messages m JOIN users u ON m.sender_id = u.id
                WHERE m.chat_id = %s
            """
            params = [chat_id]
            if before_id:
                query += " AND m.id < %s"
                params.append(before_id)
            
            query += " ORDER BY m.timestamp DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, tuple(params))
            messages = cur.fetchall()
            
            # Reverse to send in chronological order
            messages.reverse()

            message_list = [
                {
                    "id": row[0], "content": row[1], "message_type": row[2],
                    "file_name": row[3], "timestamp": row[4].isoformat(),
                    "sender_username": row[5]
                } for row in messages
            ]
            return jsonify(message_list)
    finally:
        release_db_connection(conn)

@app.route('/search_users', methods=['POST'])
@login_required
def search_users():
    query = request.json.get('query')
    user_id = session['user_id']
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username FROM users WHERE username ILIKE %s AND id != %s LIMIT 10",
                (f'%{query}%', user_id)
            )
            users = [row[0] for row in cur.fetchall()]
            return jsonify(users)
    finally:
        release_db_connection(conn)

@app.route('/create_chat', methods=['POST'])
@login_required
def create_chat():
    target_username = request.json.get('target_username')
    user_id = session['user_id']

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (target_username,))
            target_user = cur.fetchone()
            if not target_user:
                return jsonify({"success": False, "error": "User not found"}), 404
            target_user_id = target_user[0]

            # Check if a chat already exists
            cur.execute("""
                SELECT p1.chat_id FROM participants p1
                JOIN participants p2 ON p1.chat_id = p2.chat_id
                WHERE p1.user_id = %s AND p2.user_id = %s
            """, (user_id, target_user_id))
            existing_chat = cur.fetchone()

            if existing_chat:
                return jsonify({"success": True, "chat_id": existing_chat[0], "participant_username": target_username})

            # Create new chat
            cur.execute("INSERT INTO chats DEFAULT VALUES RETURNING id")
            chat_id = cur.fetchone()[0]
            cur.execute("INSERT INTO participants (user_id, chat_id) VALUES (%s, %s), (%s, %s)",
                        (user_id, chat_id, target_user_id, chat_id))
            conn.commit()

            return jsonify({"success": True, "chat_id": chat_id, "participant_username": target_username})
    except Exception as e:
        conn.rollback()
        logging.error(f"Chat creation error: {e}")
        return jsonify({"success": False, "error": "Could not create chat"}), 500
    finally:
        release_db_connection(conn)

@app.route('/upload_file/<int:chat_id>', methods=['POST'])
@login_required
def upload_file(chat_id):
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not is_user_in_chat(session['user_id'], chat_id):
        return abort(403)

    file_data = file.read()
    file_name = file.filename
    file_mime_type = file.mimetype

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (chat_id, sender_id, message_type, file_data, file_name, file_mime_type)
                VALUES (%s, %s, 'file', %s, %s, %s) RETURNING id, timestamp
                """,
                (chat_id, session['user_id'], file_data, file_name, file_mime_type)
            )
            message_id, timestamp = cur.fetchone()
            conn.commit()

            message_data = {
                "id": message_id,
                "chat_id": chat_id,
                "sender_username": session['username'],
                "message_type": "file",
                "file_name": file_name,
                "content": None,
                "timestamp": timestamp.isoformat()
            }
            _broadcast_new_message(message_data)
            return jsonify({"success": True})
    except Exception as e:
        conn.rollback()
        logging.error(f"File upload error: {e}")
        return jsonify({"success": False, "error": "Failed to upload file."}), 500
    finally:
        release_db_connection(conn)


@app.route('/files/<int:message_id>')
@login_required
def get_file(message_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chat_id, file_data, file_name, file_mime_type FROM messages WHERE id = %s",
                (message_id,)
            )
            message = cur.fetchone()
            if not message:
                return abort(404)
            
            chat_id, file_data, file_name, file_mime_type = message
            if not is_user_in_chat(session['user_id'], chat_id):
                return abort(403)

            return send_file(
                io.BytesIO(file_data),
                mimetype=file_mime_type,
                as_attachment=True,
                download_name=file_name
            )
    finally:
        release_db_connection(conn)

# --- Socket.IO Handlers ---

@socketio.on('connect')
def handle_connect():
    if 'user_id' not in session:
        logging.warning(f"Unauthenticated connection rejected: {request.sid}")
        return False  # Reject connection
    logging.info(f"Client connected: {session.get('username')} ({request.sid})")

@socketio.on('disconnect')
def handle_disconnect():
    logging.info(f"Client disconnected: {session.get('username')} ({request.sid})")

@socketio.on('join_room')
def handle_join_room(data):
    chat_id = data['chat_id']
    if is_user_in_chat(session.get('user_id'), chat_id):
        join_room(str(chat_id))
        logging.info(f"{session.get('username')} joined room {chat_id}")
    else:
        logging.warning(f"{session.get('username')} tried to join unauthorized room {chat_id}")

@socketio.on('leave_room')
def handle_leave_room(data):
    chat_id = data['chat_id']
    leave_room(str(chat_id))
    logging.info(f"{session.get('username')} left room {chat_id}")

@socketio.on('send_message')
def handle_send_message(data):
    chat_id = data['chat_id']
    content = data['content']
    sender_id = session.get('user_id')
    sender_username = session.get('username')

    if not sender_id or not is_user_in_chat(sender_id, chat_id):
        logging.warning(f"Unauthorized message attempt by {sender_username} in chat {chat_id}")
        return

    # AI Feature Logic
    if content.strip().lower().startswith('/ai '):
        if not gemini_model:
            ai_response_content = "AI feature is currently disabled."
        else:
            try:
                question = content.strip()[4:]
                response = gemini_model.generate_content(question)
                ai_response_content = response.text
            except Exception as e:
                logging.error(f"Gemini API error: {e}")
                ai_response_content = "Sorry, I couldn't process that request."
        
        ai_message = {
            "chat_id": chat_id,
            "sender_username": "Gemini AI",
            "content": ai_response_content,
            "message_type": "text",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        _broadcast_new_message(ai_message)
        return

    # Standard Message Logic
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (chat_id, sender_id, content) VALUES (%s, %s, %s) RETURNING id, timestamp",
                (chat_id, sender_id, content)
            )
            message_id, timestamp = cur.fetchone()
            conn.commit()

            message_data = {
                "id": message_id,
                "chat_id": chat_id,
                "sender_username": sender_username,
                "content": content,
                "message_type": "text",
                "timestamp": timestamp.isoformat()
            }
            _broadcast_new_message(message_data)
    except Exception as e:
        conn.rollback()
        logging.error(f"Message saving error: {e}")
    finally:
        release_db_connection(conn)


# --- Main Execution ---

if __name__ == '__main__':
    init_db()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
else:
    # When run by Gunicorn on Render, initialize DB here
    init_db_pool()
    init_db()

import os
import datetime
import io
import logging
import sys
from functools import wraps

from dotenv import load_dotenv
from flask import (Flask, request, jsonify, send_file, session)
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from psycopg2 import pool
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai

# --- Initialization & Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_super_secret_key_for_dev')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
DATABASE_URL = os.environ.get('DATABASE_URL')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
FRONTEND_URL = "https://vasiliy-katsyka.github.io"
CORS(app, supports_credentials=True, origins=[FRONTEND_URL, "http://127.0.0.1:5500", "http://localhost:5500"])
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    logging.warning("GEMINI_API_KEY not found. AI features will be disabled.")
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins=[FRONTEND_URL, "http://127.0.0.1:5500", "http://localhost:5500"])
db_pool = None

# --- Database Setup ---
def init_db_pool():
    global db_pool
    if db_pool is None:
        try:
            db_pool = pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
            logging.info("Database connection pool created.")
        except Exception as e:
            logging.error(f"Failed to create database connection pool: {e}")
            raise

def get_db_connection():
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()

def release_db_connection(conn):
    if db_pool:
        db_pool.putconn(conn)

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username VARCHAR(50) UNIQUE NOT NULL, password_hash VARCHAR(256) NOT NULL);""")
            cur.execute("""CREATE TABLE IF NOT EXISTS chats (id SERIAL PRIMARY KEY, created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW());""")
            cur.execute("""CREATE TABLE IF NOT EXISTS participants (user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE, PRIMARY KEY (user_id, chat_id));""")
            cur.execute("""CREATE TABLE IF NOT EXISTS messages (id SERIAL PRIMARY KEY, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE, sender_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, content TEXT, message_type VARCHAR(10) NOT NULL DEFAULT 'text', file_data BYTEA, file_name VARCHAR(255), file_mime_type VARCHAR(100), timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW());""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages (chat_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender_id ON messages (sender_id);")
            conn.commit()
            logging.info("Database tables checked/created.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Database initialization failed: {e}")
        raise
    finally:
        release_db_connection(conn)

# --- Decorators, Helpers, and Error Handlers ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"success": False, "error": "Authentication required."}), 401
        return f(*args, **kwargs)
    return decorated_function

def is_user_in_chat(user_id, chat_id):
    if not user_id or not chat_id:
        return False
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM participants WHERE user_id = %s AND chat_id = %s", (user_id, chat_id))
            return cur.fetchone() is not None
    except Exception as e:
        logging.error(f"Error checking user in chat: {e}")
        return False
    finally:
        release_db_connection(conn)

def _broadcast_new_message(message_data):
    try:
        chat_id = message_data.get('chat_id')
        if chat_id:
            socketio.emit('new_message', message_data, room=str(chat_id))
    except Exception as e:
        logging.error(f"Error broadcasting message: {e}", exc_info=True)

# --- API Routes ---
@app.route('/')
def index():
    return "Backend is running."

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username, password = data.get('username'), data.get('password')
    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required."}), 400
    hashed_password = generate_password_hash(password)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
        return jsonify({"success": True}), 201
    except Exception:
        conn.rollback()
        return jsonify({"success": False, "error": "Username already exists."}), 409
    finally:
        release_db_connection(conn)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username, password = data.get('username'), data.get('password')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            if user and check_password_hash(user[1], password):
                session['user_id'], session['username'] = user[0], username
                return jsonify({"success": True, "username": username})
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
    finally:
        release_db_connection(conn)

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/check_session')
@login_required
def check_session():
    return jsonify({"loggedIn": True, "username": session.get('username')})

@app.route('/get_user_chats')
@login_required
def get_user_chats():
    user_id = session.get('user_id')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                WITH last_messages AS (SELECT chat_id, content, message_type, timestamp, ROW_NUMBER() OVER(PARTITION BY chat_id ORDER BY timestamp DESC) as rn FROM messages)
                SELECT c.id as chat_id, u.username as participant_username, CASE WHEN lm.message_type = 'file' THEN 'File' ELSE lm.content END as last_message_content, lm.timestamp as last_message_timestamp
                FROM participants p_me JOIN participants p_other ON p_me.chat_id = p_other.chat_id AND p_me.user_id != p_other.user_id
                JOIN users u ON p_other.user_id = u.id JOIN chats c ON p_me.chat_id = c.id
                LEFT JOIN last_messages lm ON c.id = lm.chat_id AND lm.rn = 1
                WHERE p_me.user_id = %s ORDER BY COALESCE(lm.timestamp, c.created_at) DESC;
            """, (user_id,))
            chats = [{"chat_id": r[0], "participant_username": r[1], "last_message": r[2] or "", "timestamp": r[3]} for r in cur.fetchall()]
            return jsonify(chats)
    finally:
        release_db_connection(conn)

@app.route('/get_chat_messages/<int:chat_id>')
@login_required
def get_chat_messages(chat_id):
    if not is_user_in_chat(session.get('user_id'), chat_id):
        return jsonify({"success": False, "error": "Forbidden"}), 403
    before_id, limit = request.args.get('before', type=int), 50
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query, params = "SELECT m.id, m.content, m.message_type, m.file_name, m.timestamp, u.username as sender_username FROM messages m JOIN users u ON m.sender_id = u.id WHERE m.chat_id = %s", [chat_id]
            if before_id:
                query += " AND m.id < %s"
                params.append(before_id)
            query += " ORDER BY m.timestamp DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, tuple(params))
            messages = cur.fetchall()
            messages.reverse()
            return jsonify([{"id": r[0], "content": r[1], "message_type": r[2], "file_name": r[3], "timestamp": r[4].isoformat(), "sender_username": r[5]} for r in messages])
    finally:
        release_db_connection(conn)

@app.route('/search_users', methods=['POST'])
@login_required
def search_users():
    data = request.get_json() or {}
    query, user_id = data.get('query'), session.get('user_id')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT username FROM users WHERE username ILIKE %s AND id != %s LIMIT 10", (f'%{query}%', user_id))
            return jsonify([row[0] for row in cur.fetchall()])
    finally:
        release_db_connection(conn)

@app.route('/create_chat', methods=['POST'])
@login_required
def create_chat():
    data = request.get_json() or {}
    target_username, user_id = data.get('target_username'), session.get('user_id')
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE username = %s", (target_username,))
            target_user = cur.fetchone()
            if not target_user:
                return jsonify({"success": False, "error": "User not found"}), 404
            target_user_id = target_user[0]
            cur.execute("SELECT p1.chat_id FROM participants p1 JOIN participants p2 ON p1.chat_id = p2.chat_id WHERE p1.user_id = %s AND p2.user_id = %s", (user_id, target_user_id))
            if existing_chat := cur.fetchone():
                return jsonify({"success": True, "chat_id": existing_chat[0], "participant_username": target_username})
            cur.execute("INSERT INTO chats DEFAULT VALUES RETURNING id")
            chat_id = cur.fetchone()[0]
            cur.execute("INSERT INTO participants (user_id, chat_id) VALUES (%s, %s), (%s, %s)", (user_id, chat_id, target_user_id, chat_id))
            conn.commit()
            return jsonify({"success": True, "chat_id": chat_id, "participant_username": target_username})
    except Exception:
        conn.rollback()
        return jsonify({"success": False, "error": "Could not create chat"}), 500
    finally:
        release_db_connection(conn)

@app.route('/upload_file/<int:chat_id>', methods=['POST'])
@login_required
def upload_file(chat_id):
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({"success": False, "error": "No file selected"}), 400
    file = request.files['file']
    if not is_user_in_chat(session.get('user_id'), chat_id):
        return jsonify({"success": False, "error": "Forbidden"}), 403
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO messages (chat_id, sender_id, message_type, file_data, file_name, file_mime_type) VALUES (%s, %s, 'file', %s, %s, %s) RETURNING id, timestamp", (chat_id, session.get('user_id'), file.read(), file.filename, file.mimetype))
            message_id, timestamp = cur.fetchone()
            conn.commit()
            _broadcast_new_message({"id": message_id, "chat_id": chat_id, "sender_username": session.get('username'), "message_type": "file", "file_name": file.filename, "content": None, "timestamp": timestamp.isoformat()})
            return jsonify({"success": True})
    except Exception:
        conn.rollback()
        return jsonify({"success": False, "error": "Failed to upload file."}), 500
    finally:
        release_db_connection(conn)

@app.route('/files/<int:message_id>')
@login_required
def get_file(message_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT chat_id, file_data, file_name, file_mime_type FROM messages WHERE id = %s", (message_id,))
            message = cur.fetchone()
            if not message:
                return jsonify({"success": False, "error": "File not found"}), 404
            chat_id, file_data, file_name, file_mime_type = message
            if not is_user_in_chat(session.get('user_id'), chat_id):
                return jsonify({"success": False, "error": "Forbidden"}), 403
            return send_file(io.BytesIO(file_data), mimetype=file_mime_type, as_attachment=True, download_name=file_name)
    finally:
        release_db_connection(conn)

# --- Socket.IO Handlers ---
@socketio.on('connect')
def handle_connect():
    try:
        if 'user_id' not in session:
            logging.warning(f"Unauthenticated socket connection rejected: {request.sid}")
            return False
        logging.info(f"Client connected: {session.get('username')} ({request.sid})")
    except Exception as e:
        logging.error(f"[connect] Unhandled exception: {e}", exc_info=True)
        return False

@socketio.on('disconnect')
def handle_disconnect():
    try:
        logging.info(f"Client disconnected: {session.get('username', 'Unknown')} ({request.sid})")
    except Exception as e:
        logging.error(f"[disconnect] Unhandled exception: {e}", exc_info=True)

@socketio.on('join_room')
def handle_join_room(data):
    try:
        chat_id = data.get('chat_id') if isinstance(data, dict) else None
        if not chat_id:
            logging.warning(f"Join room request missing chat_id from sid {request.sid}")
            return
        if 'user_id' in session and is_user_in_chat(session.get('user_id'), chat_id):
            join_room(str(chat_id))
            logging.info(f"{session.get('username')} joined room {chat_id}")
    except Exception as e:
        logging.error(f"[join_room] Unhandled exception for sid {request.sid}: {e}", exc_info=True)

@socketio.on('leave_room')
def handle_leave_room(data):
    try:
        chat_id = data.get('chat_id') if isinstance(data, dict) else None
        if not chat_id:
            return
        if 'user_id' in session:
            leave_room(str(chat_id))
            logging.info(f"{session.get('username')} left room {chat_id}")
    except Exception as e:
        logging.error(f"[leave_room] Unhandled exception for sid {request.sid}: {e}", exc_info=True)

@socketio.on('send_message')
def handle_send_message(data):
    try:
        data = data if isinstance(data, dict) else {}
        chat_id = data.get('chat_id')
        content = data.get('content')
        if 'user_id' not in session or not chat_id or content is None or not is_user_in_chat(session.get('user_id'), chat_id):
            return
        sender_id = session.get('user_id')
        if content.strip().lower().startswith('/ai '):
            try:
                question = content.strip()[4:]
                ai_response_content = "AI feature is currently disabled." if not gemini_model else gemini_model.generate_content(question).text
            except Exception as ai_e:
                logging.error(f"Gemini API error: {ai_e}")
                ai_response_content = "Sorry, I couldn't process that request."
            _broadcast_new_message({"chat_id": chat_id, "sender_username": "Gemini AI", "content": ai_response_content, "message_type": "text", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            return
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO messages (chat_id, sender_id, content) VALUES (%s, %s, %s) RETURNING id, timestamp", (chat_id, sender_id, content))
                message_id, timestamp = cur.fetchone()
                conn.commit()
                _broadcast_new_message({"id": message_id, "chat_id": chat_id, "sender_username": session.get('username'), "content": content, "message_type": "text", "timestamp": timestamp.isoformat()})
        except Exception as db_e:
            conn.rollback()
            logging.error(f"Database error on send_message: {db_e}")
        finally:
            release_db_connection(conn)
    except Exception as e:
        logging.error(f"[send_message] Unhandled exception for sid {request.sid}: {e}", exc_info=True)

# --- Main Execution & CLI ---
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'initdb':
        init_db_pool()
        init_db()
        print("Database initialized.")
    else:
        init_db_pool()
        init_db() # Also init for local dev convenience
        print("Starting development server...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
else:
    init_db_pool() # For Gunicorn on Render

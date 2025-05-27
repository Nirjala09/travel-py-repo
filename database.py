import sqlite3
import hashlib

# Connect to SQLite database (creates the database file if not exists)
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# Create a table to store user data
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
''')
conn.commit()

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to add a new user (Signup)
def add_user(username, password):
    try:
        hashed_password = hash_password(password)
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists

# Function to check user credentials (Login)
def check_user(username, password):
    hashed_password = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    return cursor.fetchone() is not None
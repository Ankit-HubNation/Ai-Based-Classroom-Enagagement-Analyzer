import sqlite3
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

conn = sqlite3.connect('database.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
''')

hashed_pw = bcrypt.generate_password_hash("admin123").decode('utf-8')

c.execute("INSERT OR IGNORE INTO users (username,password,role) VALUES (?,?,?)",
          ("admin", hashed_pw, "admin"))

conn.commit()
conn.close()

print("Database Created Successfully!")
import sqlite3

db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)''')

data = [
    ('user1', 'password1'),
    ('user2', 'password2'),
    ('user3', 'pas1'),
    ('user4', 'pass2'),
    ('ninel', 'benyush'),
    ('morin', 'lugasi'),
    ('nina', 'benyush'),
    ('morinn', 'lugasi')
]

for username, password in data:
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    except sqlite3.IntegrityError:
        print(f"Username '{username}' already exists.")


cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(f"id: {row[0]}, username: '{row[1]}', password: '{row[2]}'")


conn.commit()
conn.close()

print("New database initialized successfully.")

import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('users.db')
cursor = conn.cursor()

# Create the users table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)''')

# Sample data for insertion
data = [
    ('ninel', 'benyush'),
    ('morin', 'lugasi')
]

# Insert the data into the users table, handling duplicates
for username, password in data:
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    except sqlite3.IntegrityError:
        print(f"Username '{username}' already exists. Skipping...")

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database initialized successfully.")

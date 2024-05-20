import sqlite3

# Define the path to the new database
db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'

# Connect to the new SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the users table if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL
)''')

# Sample data for insertion
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

# Insert the data into the users table, handling duplicates
for username, password in data:
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    except sqlite3.IntegrityError:
        print(f"Username '{username}' already exists.")

# Verify the inserted data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(f"id: {row[0]}, username: '{row[1]}', password: '{row[2]}'")

# Commit changes and close the connection
conn.commit()
conn.close()

print("New database initialized successfully.")

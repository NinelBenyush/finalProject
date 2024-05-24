import sqlite3

db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

new_column_name = 'email'
cursor.execute(f"ALTER TABLE users ADD COLUMN {new_column_name} TEXT")

cursor.execute("SELECT id, username FROM users")
rows = cursor.fetchall()
for row in rows:
    user_id = row[0]
    username = row[1]
    email = f"{username}@example.com"  # Generate email based on username
    cursor.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))

conn.commit()

# Verify the inserted data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the connection
conn.close()
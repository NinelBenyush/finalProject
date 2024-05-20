import sqlite3

# Define the path to the new database
db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/new_users.db'

# Connect to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Verify the inserted data
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()
for row in rows:
    print(f"id: {row[0]}, username: '{row[1]}', password: '{row[2]}'")

# Close the connection
conn.close()

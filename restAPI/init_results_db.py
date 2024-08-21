#create a db that holds the results
import sqlite3
db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fileName TEXT NOT NULL
)''')

data = [
    ('sales2023'),
]

for fileName in data:
    try:
        cursor.execute("INSERT INTO results (fileName) VALUES (?)", (fileName,))
    except sqlite3.IntegrityError:
        print("error")


cursor.execute("SELECT * FROM results")
rows = cursor.fetchall()
for row in rows:
    print(f"name: {row[0]}")

conn.commit()
conn.close()

print("New database initialized successfully.")
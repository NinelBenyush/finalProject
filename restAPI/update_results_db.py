import sqlite3
from datetime import datetime

db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/results.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''PRAGMA table_info(results)''')
columns = [col[1] for col in cursor.fetchall()]
if 'date' not in columns:
    cursor.execute('''ALTER TABLE results ADD COLUMN date TEXT''')


data = [
    ('sales2023', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
]

for fileName, date in data:
    try:
        cursor.execute("INSERT INTO results (fileName, date) VALUES (?, ?)", (fileName, date))
    except sqlite3.IntegrityError:
        print("Error inserting data")

cursor.execute("SELECT * FROM results")
rows = cursor.fetchall()
for row in rows:
    print(f"id: {row[0]}, fileName: {row[1]}, date: {row[2]}")

conn.commit()
conn.close()

print("Database updated successfully.")

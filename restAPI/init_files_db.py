import sqlite3

db_path = 'C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/uploaded_files.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS uploaded_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    description TEXT NOT NULL
)''')

data = [
    ('book1.xlsx','2023 sales'),
]

for file_name, description in data:
    try:
        cursor.execute("INSERT INTO uploaded_files (file_name, description) VALUES (?, ?)", (file_name, description))
    except sqlite3.IntegrityError:
        print("")

cursor.execute("SELECT * FROM uploaded_files")
rows = cursor.fetchall()
for row in rows:
    print(f"name: {row[0]}, des: '{row[1]}'")


conn.commit()
conn.close()

print("New database initialized successfully.")
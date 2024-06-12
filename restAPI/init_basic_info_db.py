import sqlite3

db_path = "C:/Users/Nina/Desktop/finalProject/finalProjectWebsite/restAPI/basicInfo.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if the emailAddress column already exists
cursor.execute("PRAGMA table_info(basicInfo)")
columns = [column[1] for column in cursor.fetchall()]
if "emailAddress" not in columns:
    cursor.execute("ALTER TABLE basicInfo ADD COLUMN emailAddress TEXT")

# Optionally, update existing rows with a default email address
cursor.execute("UPDATE basicInfo SET emailAddress = 'default@example.com' WHERE emailAddress IS NULL")

# Create the table if it doesn't exist already
cursor.execute('''CREATE TABLE IF NOT EXISTS basicInfo (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    firstName TEXT NOT NULL,
    lastName TEXT NOT NULL,
    companyName TEXT NOT NULL,
    phoneNumber TEXT NOT NULL UNIQUE,
    companyDescription TEXT NOT NULL,
    emailAddress TEXT
)''')

# Insert new data
data = [
    ('Ninel', 'Benyush', 'example company', '0535353533', 'exampleee', 'example@example.com'),
    ('nine', 'eee', 'eee', '7777', '7777', 'another@example.com')
]

for firstName, lastName, companyName, phoneNumber, companyDescription, emailAddress in data:
    try:
        cursor.execute("INSERT INTO basicInfo (firstName, lastName, companyName, phoneNumber, companyDescription, emailAddress) VALUES (?, ?, ?, ?, ?, ?)", 
                       (firstName, lastName, companyName, phoneNumber, companyDescription, emailAddress))
    except sqlite3.IntegrityError:
        print('error')

cursor.execute("SELECT * FROM basicInfo")
rows = cursor.fetchall()
for row in rows:
    print(f"id: {row[0]}, firstName: '{row[1]}', lastName: '{row[2]}', companyName: '{row[3]}', phoneNumber: '{row[4]}', companyDescription: '{row[5]}', emailAddress: '{row[6]}'")

conn.commit()
conn.close()

print("basicInfo database initialized successfully.")

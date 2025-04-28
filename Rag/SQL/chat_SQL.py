import sqlite3

# Connect to the database
conn = sqlite3.connect('students.db')
cursor = conn.cursor()

# Create the table if it doesn't exist
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT (
    name TEXT,
    course TEXT,
    grade TEXT,
    marks INTEGER,
    PRIMARY KEY (name, course)
)
"""
cursor.execute(table_info)

# ❗One-time cleanup: clear the table to remove old duplicate entries
cursor.execute("DELETE FROM STUDENT")

# Insert records only if they don't already exist
students = [
    ('Krish', 'Data Science', 'A', 90),
    ('John', 'Data Science', 'B', 100),
    ('Mukesh', 'Data Science', 'A', 86),
    ('Jacob', 'DEVOPS', 'A', 50),
    ('Dipesh', 'DEVOPS', 'A', 35),
]

for student in students:
    cursor.execute("INSERT OR IGNORE INTO STUDENT VALUES (?, ?, ?, ?)", student)

# Fetch and display the records
cursor.execute("SELECT * FROM STUDENT")
records = cursor.fetchall()

print("The inserted records are")
for record in records:
    print(record)

# Commit and close
conn.commit()
conn.close()

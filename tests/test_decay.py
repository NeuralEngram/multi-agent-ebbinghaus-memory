import sqlite3, os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "memory.db")
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    DELETE FROM episodes 
    WHERE context LIKE '%alice%'
    AND (content LIKE '%?%' 
    OR content LIKE '%Summarize%'
    OR content LIKE '%What %'
    OR content LIKE '%Tell me%'
    OR content LIKE '%Remind%'
    OR content LIKE '%Who am%')
""")
conn.commit()

count = conn.execute("SELECT COUNT(*) FROM episodes WHERE context LIKE '%alice%'").fetchone()[0]
print(f"Alice episodes after cleanup: {count}")
conn.close()
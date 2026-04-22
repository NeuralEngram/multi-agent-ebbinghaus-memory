import sqlite3

conn = sqlite3.connect("memory.db")
conn.execute("DELETE FROM episodes WHERE context LIKE '%alice%'")
conn.execute("DELETE FROM episodes WHERE context LIKE '%bob%'")
conn.execute("DELETE FROM episodes WHERE context LIKE '%charlie%'")
conn.commit()
conn.close()
print("Done! All wiped.")
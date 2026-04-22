import sqlite3
import json

conn = sqlite3.connect("memory.db")
rows = conn.execute("SELECT episode_id, content, context, stability_hours, review_count FROM episodes").fetchall()
conn.close()

print(f"Total episodes: {len(rows)}")
for r in rows:
    print(f"\n  content : {r[1]}")
    print(f"  context : {r[2]}")
    print(f"  stability: {r[3]:.1f}h  reviews: {r[4]}")
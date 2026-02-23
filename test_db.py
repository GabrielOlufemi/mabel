# test_db.py
from app.db.database import engine, init_db
from sqlalchemy import text

def test():
    init_db()
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ DB connected:", result.fetchone())
        
        # also confirm the users table was created
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]
        print("✅ Tables found:", tables)

test()
import mysql.connector
import os

def get_db_connection():
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        user="app_user",
        password="Bottle80%40",
        database="booking_system",
        autocommit=True
    )
    conn.start_transaction()
    return conn

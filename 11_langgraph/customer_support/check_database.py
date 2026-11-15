import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect("Chinook.db")

# Function to save table contents to CSV
def save_table_to_csv(table_name):
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql_query(query, conn)
    csv_file_name = f"{table_name}.csv"
    df.to_csv(csv_file_name, index=False)
    print(f"Saved {table_name} to {csv_file_name}")

# List of table names to save
table_names = [
    'Album',
    'Artist',
    'Customer',
    'Employee',
    'Genre',
    'Invoice',
    'InvoiceLine',
    'MediaType',
    'Playlist',
    'PlaylistTrack',
    'Track'
]

# Save the contents of each table to a CSV file
for table_name in table_names:
    save_table_to_csv(table_name)

# Close the connection
conn.close()


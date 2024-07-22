import os
import json
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
COLLECTION_NAME = os.getenv("POKEMON_COLLECTION_NAME")

# Connect to MongoDB
client = pymongo.MongoClient(CONNECTION_STRING)
print("Connected to MongoDB")

# Create a database and a collection
db = client.pokedex

# Load data from CSV
try:
    data = pd.read_csv("../data/pokedex.csv")
    print("Data loaded")
except FileNotFoundError:
    print("File not found")

#Convert CSV data to JSON
payload = json.loads(data.to_json(orient='records'))

#Collection count
count = db[COLLECTION_NAME].count_documents({})

# Check if collection is not empty
if count > 0:
    print("Collection contains", count, "documents")

    # Clear the collection
    db[COLLECTION_NAME].drop()
    print("Collection cleared")

# Insert data into MongoDB
db[COLLECTION_NAME].insert_many(payload)
print("Data inserted")

client.close()
from openai import AzureOpenAI
import pymongo
import os
import json
import time
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

load_dotenv()

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION")
AOAI_KEY = os.getenv("AOAI_KEY")
COLLECTION_NAME = os.getenv("POKEMON_COLLECTION_NAME")
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("EMBEDDINGS_DEPLOYMENT_NAME")

# Load the Azure OpenAI API key
ai_client = AzureOpenAI(
    azure_endpoint = AOAI_ENDPOINT,
    api_version = AOAI_API_VERSION,
    api_key = AOAI_KEY
)

client = pymongo.MongoClient(CONNECTION_STRING)
print("Connected to MongoDB")

db = client.pokedex

# Generate embeddings
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def generate_embeddings(text: str):
    response = ai_client.embeddings.create(input=text, model=EMBEDDINGS_DEPLOYMENT_NAME)
    embeddings = response.data[0].embedding
    
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI
    
    return embeddings

# Add a vector field to the collection
def add_collection_content_vector_field(collection_name: str):
    collection = db[collection_name]
    bulk_operations = []
    
    for doc in collection.find({}, batch_size=3):
        # remove any previous contentVector embeddings
        if "contentVector" in doc:
            del doc["contentVector"]

        # generate embeddings for the document string representation
        content = json.dumps(doc, default=str)
        content_vector = generate_embeddings(content)       
        
        bulk_operations.append(pymongo.UpdateOne(
            {"_id": doc["_id"]},
            {"$set": {"contentVector": content_vector}},
            upsert=True
        ))

    # execute bulk operations
    collection.bulk_write(bulk_operations)

# Add the contentVector field to the pokemon collection
add_collection_content_vector_field(COLLECTION_NAME)

# Create a vector index on the contentVector field
db.command({
  'createIndexes': 'pokemon',
  'indexes': [
    {
      'name': 'VectorSearchIndex',
      'key': {
        "contentVector": "cosmosSearch"
      },
      'cosmosSearchOptions': {
        'kind': 'vector-ivf',
        'numLists': 1,
        'similarity': 'COS',
        'dimensions': 1536
      }
    }
  ]
})

client.close()
ai_client.close()
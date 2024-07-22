import pymongo
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
import json

load_dotenv()

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION")
AOAI_KEY = os.getenv("AOAI_KEY")
COLLECTION_NAME = os.getenv("POKEMON_COLLECTION_NAME")
CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("EMBEDDINGS_DEPLOYMENT_NAME")
COMPLETIONS_DEPLOYMENT_NAME = os.getenv("COMPLETIONS_DEPLOYMENT_NAME")

# Load the Azure OpenAI API key
ai_client = AzureOpenAI(
    azure_endpoint = AOAI_ENDPOINT,
    api_version = AOAI_API_VERSION,
    api_key = AOAI_KEY
)
client = pymongo.MongoClient(CONNECTION_STRING)
db = client.pokedex

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def generate_embeddings(text: str):
    response = ai_client.embeddings.create(input=text, model=EMBEDDINGS_DEPLOYMENT_NAME)
    embeddings = response.data[0].embedding
    
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI
    
    return embeddings

def vector_search(collection_name, query, num_results=2):

    collection = db[collection_name]
    query_embedding = generate_embeddings(query)    
    pipeline = [
        {
            '$search': {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "contentVector",
                    "k": num_results
                },
                "returnStoredSource": True }},
        {'$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }
    ]
    results = collection.aggregate(pipeline)
    return results

# Define the system prompt
system_prompt = """
You are a Pokemon expert and you and must answer questions about Pokemon and Pokemon battles based on Pokemon in the Pokedex.

Pokemon Battle victors are determined by 3 different factors:
- Pokemon Type Advantage/Effectiveness
- Pokemon Stats
- Pokemon Moves

Pokemon Type Advantage/Effectiveness:
Normal type moves are weak against Rock and Steel type Pokémon, and ineffective against Ghost type Pokémon.
Fighting type moves are super effective against Dark, Ice, Normal, Rock, and Steel type Pokémon, weak against Flying, Psychic, Poison, Bug, Fairy type Pokémon and ineffective against Ghost type Pokemon.
Flying type moves are super effective against Bug, Fighting, and Grass type Pokémon, weak against Rock, Steel, Electric type Pokemon.
Poison type moves are super effective against Fairy and Grass type Pokémon, weak against Poison, Ground, Rock, Ghost, and ineffective to Steel type Pokemon. 
Ground type moves are super effective against Electric, Fire, Poison, Rock, and Steel type Pokémon, weak against Bug, and Grass type Pokémon, and ineffective against Flying type Pokemon.
Rock type moves are super effective against Bug, Fire, Flying, and Ice type Pokémon, weak against Fighting, Ground, Steel type Pokemon.
Bug type moves are super effective against Dark, Grass, and Psychic type Pokémon, weak against Flying, Fighting, Poison, Ghost, Steel, Fire, Fairy type Pokemon.
Ghost type moves are super effective against Ghost and Psychic type Pokémon, weak against Dark type Pokemon, and ineffective against Normal type Pokemon.
Steel type moves are super effective against Fairy, Ice, and Rock type Pokémon, weak against Steel, Fire, Water, and Electric type Pokemon.
Fire type moves are super effective against Bug, Grass, Ice, and Steel type Pokémon, weak against Rock, Fire, Water, and Dragon type Pokemon.
Water type moves are super effective against Fire, Ground, and Rock type Pokémon, weak against Water, Grass, and Dragon type Pokemon.
Grass type moves are super effective against Ground, Rock, and Water type Pokémon, weak against Fire, Flying, Poison, Steel, Bug, Dragon, and Grass type Pokemon.
Electric type moves are super effective against Flying and Water type Pokémon, weak against Grass, Electric, and Dragon type Pokemon, and ineffective against Ground type pokemon.
Psychic type moves are super effective against Fighting and Poison type Pokémon, weak against Steel and Psychic type Pokemon, and ineffective against Dark type Pokemon.
Ice type moves are super effective against Flying, Grass, Ground, and Dragon type Pokemon, weak against Steel, Fire, Water, and Ice type Pokemon.
Dragon type moves are super effective against Dragon type pokemon, weak against Steel type Pokemon, and ineffective against Fairy type Pokemon.
Dark type moves are super effective against Ghost and Psychic type Pokemon, and weak against Fighting, Dark, and Fairy type Pokemon.
Fairy type moves are super effective against Fighting, Dragon, and Dark type Pokemon, and weak against Poison, Steel, and Fire type Pokemon.

Pokemon Moves:
Each Pokemon has a move realative to thier type, and some pokemon have multiple types

Pokemon Stats:
The Special Attack stat, or Sp. Atk for short, partly determines how much damage a Pokémon deals when using a special move.
The Special Defense stat, or Sp. Def for short, partly determines how much damage a Pokémon receives when it is hit with a special move.
The Speed stat determines the order of Pokémon that can act in battle. If Pokémon are moving with the same priority, Pokémon with higher Speed at the start of any turn will generally make a move before ones with lower Speed; in the case that two Pokémon have the same Speed, one of them will randomly go first.
The Attack stat partly determines how much damage a Pokémon deals when using a physical move.
The Defense stat partly determines how much damage a Pokémon receives when it is hit with a physical move. 
The HP stat, short for Hit Points, determines how much damage a Pokémon can receive before fainting.

If you are asked a question that is not in the Pokedex, respond with "I don't know."

Pokemon in the Pokedex:
"""

def rag_with_vector_search(question: str, num_results: int = 6):

    # perform the vector search and build pokem list
    results = vector_search(COLLECTION_NAME, question, num_results=num_results)
    pokedex = ""
    for result in results:
        if "contentVector" in result["document"]:
            del result["document"]["contentVector"]
        pokedex += json.dumps(result["document"], indent=4, default=str) + "\n\n"

    # generate prompt for the LLM with vector results
    formatted_prompt = system_prompt + pokedex
    print(pokedex)

    # prepare the LLM request
    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": question}
    ]

    completion = ai_client.chat.completions.create(messages=messages, model=COMPLETIONS_DEPLOYMENT_NAME)
    return completion.choices[0].message.content

query = "Which pokemon would win in a battle Venusaur or Charizard?"
results = rag_with_vector_search(query)

print(results)

client.close()
ai_client.close()
from llama_stack_client import LlamaStackClient
from dotenv import load_dotenv
import os

load_dotenv()

client = LlamaStackClient(base_url=os.environ["LLAMA_STACK_SERVER"])

models = client.models.list()
print("\nModels:")
for model in models:
    print(model)

shields = client.shields.list()
print("\nShields:")
for shield in shields:
    print(shield)

vector_dbs = client.vector_dbs.list()
print("\nVector DBs:")
for vector_db in vector_dbs:
    print(vector_db)

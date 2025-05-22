import os
import uuid

from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient

from utility_scripts.dev_conf_cz_2025_info import fetch_dev_conf_talks_md

load_dotenv()

vector_db_id = "dev_conf_cz_info"
client = LlamaStackClient(base_url=os.getenv("LLAMA_STACK_SERVER"))

vector_dbs = client.vector_dbs.list()
print(f"Configured Vector DBs: {vector_dbs}")
for vector_db in vector_dbs:
    client.vector_dbs.unregister(vector_db.identifier)

# client.vector_dbs.unregister(vector_db_id=vector_db_id)
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    provider_id="faiss",
    embedding_model=os.getenv("EMBEDDING_MODEL_ID"),
    embedding_dimension=384

)

client.tool_runtime.rag_tool.insert(documents=fetch_dev_conf_talks_md(),
                                    vector_db_id=vector_db_id,
                                    chunk_size_in_tokens=512)

rag_results = client.tool_runtime.rag_tool.query(
    content="Find all talks on llama stack",
    vector_db_ids=[vector_db_id],
)

print(rag_results)

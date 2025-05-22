import os

from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.shared_params import UserMessage, Document, SystemMessage


load_dotenv()

vector_db_id = "dev_conf_cz_info"
client = LlamaStackClient(base_url=os.getenv("LLAMA_STACK_SERVER"))

def chat_with_rag_enabled_inference(query: str) -> str:
    rag_results = client.tool_runtime.rag_tool.query(
        vector_db_ids=[vector_db_id],
        content=query,
    )

    print("\n\n")
    print(rag_results)

    context = [chunk.text for chunk in rag_results.content]

    messages = [
        UserMessage(role="user", content=query),
        SystemMessage(role="system",
                      content="You are a helpful assistant. Use the provided context to answer accurately. \nContext: \n" + "\n".join(
                          context)),

    ]

    response = client.inference.chat_completion(
        model_id=os.getenv("INFERENCE_MODEL_ID"),
        messages=messages,
    )

    return response.completion_message.content

if __name__ == "__main__":
    chat_with_rag_enabled_inference("Find talks by Abhishek Kumar")
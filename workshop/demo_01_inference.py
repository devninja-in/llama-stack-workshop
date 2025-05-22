from llama_stack_client import *
import os
from dotenv import load_dotenv
from llama_stack_client.types.shared_params import UserMessage, SystemMessage

load_dotenv()


client = LlamaStackClient(base_url=os.getenv("LLAMA_STACK_SERVER"))

def chat_completion_with_inference(content: str):
    response = client.inference.chat_completion(
        model_id=os.getenv("INFERENCE_MODEL_ID"),
        messages=[
            SystemMessage(role="system", content="You're a helpful assistant."),
            UserMessage(role="user", content=content),
        ],
        stream=False,
    )
    print(f"Content={content}, response={response}")
    return response.completion_message.content

if __name__ == "__main__":
    print(chat_completion_with_inference("Tell me a short story about you."))
import uuid

from llama_stack_client import LlamaStackClient, Agent
from dotenv import load_dotenv
import os

from llama_stack_client.types import UserMessage

load_dotenv()

client = LlamaStackClient(base_url=os.getenv("LLAMA_STACK_SERVER"))

agent = Agent(
    client=client,
    model=os.getenv("INFERENCE_MODEL_ID"),
    instructions="You are a helpful assistant.",
    input_shields=[os.getenv("SHIELD_ID")],
    output_shields=[os.getenv("SHIELD_ID")],
)


def create_simple_agent_session(prefix):
    session_id = agent.create_session(f"{prefix}-{uuid.uuid4()}")
    print(f"Created session id: {session_id}")
    return session_id


def chat_with_simple_agent(session_id, query):
    response = agent.create_turn(
        session_id=session_id,
        messages=[
            UserMessage(
                role="user", content=query
            )
        ],
        stream=False,
    )
    print(f"Query: {query}, response: {response}")
    return response.output_message.content


if __name__ == "__main__":
    print(chat_with_simple_agent(create_simple_agent_session("demo_agent_session"),
                          "Tell me about you and explain how you are different from simple chat completion"))

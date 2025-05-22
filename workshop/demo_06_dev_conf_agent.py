import os
import uuid

from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient, Agent
from llama_stack_client.types import UserMessage
from llama_stack_client.types.agents.turn_create_params import ToolgroupAgentToolGroupWithArgs

load_dotenv()

client = LlamaStackClient(base_url=os.getenv('LLAMA_STACK_SERVER'))

vector_db_id = "dev_conf_cz_info"

agent = Agent(
    client=client,
    model=os.getenv("INFERENCE_MODEL_ID"),
    instructions="You are a helpful Dev Conf CZ assistant. Use RAG tool to fetch context and provide response to user query. If context is not available then use RAG tool.",
    input_shields=[os.getenv("SHIELD_ID")],
    output_shields=[os.getenv("SHIELD_ID")],
    tools=[ToolgroupAgentToolGroupWithArgs(
        name="builtin::rag/knowledge_search",
        args={
            "vector_db_ids": [vector_db_id]
        }
    )],
)

def create_dev_conf_cz_agent_session(prefix: str):
    session_id = agent.create_session(f"{prefix}-{uuid.uuid4()}")
    print(f"session_id: {session_id}")
    return session_id

def chat_with_dev_conf_cz_agent(session_id: str, query: str):
    response = agent.create_turn(
        session_id=session_id,
        messages=[UserMessage(role="user", content=query)],
        stream=False
    )
    print(f"query: {query}, response: {response}")
    return response.output_message.content

if __name__ == "__main__":
    chat_with_dev_conf_cz_agent(create_dev_conf_cz_agent_session("demo_session"), "Find all LLama Stack talk in Dev Conf CZ with full details")
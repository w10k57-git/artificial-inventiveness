from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

agent = Agent(
    "ollama:gemma3:4b",
    instructions="Be concise, reply with one sentence.",
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)

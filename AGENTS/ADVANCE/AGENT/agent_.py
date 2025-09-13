from agents import Agent, function_tool, Runner , RunConfig , OpenAIChatCompletionsModel , set_tracing_disabled
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
set_tracing_disabled(disabled=True)
API_KEY=os.environ.get("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=client,
)

config = RunConfig(
    model=model,
    model_provider=client
)

@function_tool
def greet_user(name:str)->str:
    return f"Hello {name}"


Greeting_agent = Agent(
    name="Greeting Agent",
    instructions="Greet the user by name.",
    tools=[greet_user]
)

output = Runner.run_sync(Greeting_agent,"My name is hamza.",run_config=config)
print(f"Final Output: {output.final_output}")

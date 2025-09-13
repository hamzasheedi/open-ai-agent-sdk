from agents import Agent, function_tool, Runner , RunConfig , OpenAIChatCompletionsModel , set_tracing_disabled,RunContextWrapper , enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
import os

enable_verbose_stdout_logging()
load_dotenv()
set_tracing_disabled(disabled=True)
API_KEY=os.environ.get("GEMINI_API_KEY")


@dataclass
class UserContext:
    name: str
    uid: str
    is_pro_user: bool

    async def fetch_purchases(self):
        # Simulate async DB/API call
        return ["item1", "item2"]

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

config = RunConfig(
    model=model,
    model_provider=client
)


user_context = UserContext(name="hamza",uid=123,is_pro_user=True)


# Jab agent tools fetch karta hai, toh run_context pass hota hai.
# run_context mein aapka context object hota hai (e.g. user info).
# Tools ko enable/disable karne ke liye bhi context use hota hai.



@function_tool
def get_user_details(user_context:RunContextWrapper[UserContext]):
    user = user_context.context  
    return user  


# Generic[TContext] ka matlab hai yeh class kisi bhi context type ke sath kaam kar sakti hai.
# TContext aap define karte ho (e.g. UserContext, SessionContext).
# Is se type safety milti hai, aur tools/agents ko strongly-typed context milta hai.


agent = Agent[UserContext](
    name="Context Agent",
    instructions="Greet the user by name and tell if they are pro.",
    tools=[get_user_details],
)


# Jab bhi agent run hota hai, context pass hota hai.
# Yeh context tools, handoffs, instructions, sab ko milta hai.

output = Runner.run_sync(agent,"Hello.",context=user_context,run_config=config)
print(f"Final Output: {output.final_output}")

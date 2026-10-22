import sys
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

print("1. Connecting to your local llama3.2 Chat model...")
# Initialize the local Ollama connection
model = ChatOllama(model="llama3.2", temperature=0)

# Create a custom Python code tool for the agent
@tool
def count_letters(word: str) -> str:
    """Use this tool when you need to count the exact number of characters or letters in a word."""
    length = len(word)
    return f"The word '{word}' has exactly {length} characters."

tools = [count_letters]

print("2. Assembling the modern local Agent architecture...")
# Modern LangChain agents require an explicit prompt template container with a placeholder for the agent scratchpad
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise local AI engineer assistant. Use your tools whenever necessary to calculate accurate answers."),
    ("placeholder", "{chat_history}"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Construct the native tool-calling execution framework
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("3. Asking the agent to solve a problem using local tool execution...")
try:
    response = agent_executor.invoke({
        "input": "How many letters are in the word 'langchain'?"
    })

    print("\n[SUCCESS] Agent Output:")
    print(response["output"])

except Exception as runtime_fault:
    print(f"\n🚨 Execution Fault: {runtime_fault}")

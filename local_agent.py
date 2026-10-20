from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

print("1. Connecting to your local llama3.2 Chat model...")
# Using ChatOllama allows the model to natively use tools (bind_tools)
model = ChatOllama(model="llama3.2")

# Create a custom Python code tool for the agent
@tool
def count_letters(word: str) -> str:
    """Use this tool when you need to count the exact number of characters or letters in a word."""
    length = len(word)
    return f"The word '{word}' has exactly {length} characters."

tools = [count_letters]

print("2. Assembling the modern local Agent...")
agent = create_agent(model=model, tools=tools)

print("3. Asking the agent to solve a problem using Python...")
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "How many letters are in the word 'langchain'?"}
    ]
})

print("\n[SUCCESS] Agent Output:")
# Print the final message content from the agent's history
print(response["messages"][-1].content)

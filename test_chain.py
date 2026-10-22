import sys
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("Connecting to your local llama3.2 model...")
# Setting temperature to 0.7 gives the joke generation a bit more creative flair
model = OllamaLLM(model="llama3.2", temperature=0.7)

prompt = ChatPromptTemplate.from_template("Tell me a 1-sentence joke about {topic}")
chain = prompt | model | StrOutputParser()

print("Sending request to Ollama...")
try:
    print("\n[SUCCESS] Model Response: ", end="", flush=True)
    
    # Using chain.stream instead of invoke to read tokens in real-time
    for chunk in chain.stream({"topic": "programming"}):
        print(chunk, end="", flush=True)
        
    print() # Print a final newline
except Exception as e:
    print(f"\n[ERROR] Could not complete chain: {e}")

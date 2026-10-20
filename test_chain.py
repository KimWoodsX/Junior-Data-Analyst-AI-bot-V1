from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("Connecting to your local llama3.2 model...")
model = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_template("Tell me a 1-sentence joke about {topic}")
chain = prompt | model | StrOutputParser()

print("Sending request to Ollama...")
try:
    response = chain.invoke({"topic": "programming"})
    print("\n[SUCCESS] Model Response:")
    print(response)
except Exception as e:
    print(f"\n[ERROR] Could not complete chain: {e}")

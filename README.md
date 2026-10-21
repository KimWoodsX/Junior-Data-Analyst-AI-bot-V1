# 🤖 Junior Data Analyst AI Bot (V1)
https://junior-data-analyst-ai-bot-v1-9nmiy7zcng8nt2eh9xzl6j.streamlit.app/
An enterprise-grade, localized Data Science & Analytics orchestration framework built using **LangChain Expression Language (LCEL)** and **Streamlit**. This agent is engineered to process local tabular data, generate statistical visualizations, and execute stateful conversations using fully local AI inference.

---

## 🧠 Architecture & Key Breakthroughs

This project serves as a production-ready showcase of modern LLM engineering paradigms, moving away from rigid hardcoded logic into flexible agentic workflows:

*   **LangChain Expression Language (LCEL) Pipeline:** Architecture relies on a modular assembly line mapped out via the pipe operator (`|`):
    `ChatPromptTemplate ──> ChatOllama (The Brain) ──> StrOutputParser`
*   **Intelligent Tool Calling (`@tool`):** Solves the inherent mathematical and logical blindspots of language models. By binding native Python logic (`pandas`, `numpy`) directly to the model via decorators, the agent autonomously calls the execution engine when complex calculations are required.
*   **Stateful Contextual Memory:** Implements conversation histories tracking `system`, `user`, and `assistant` states. This enables the local model to retain contextual thread awareness across multi-turn interactions.
*   **Headless Graphic Context Sandboxing:** The analytical engine leverages `matplotlib` and `seaborn` forced into headless execution mode (`matplotlib.use('Agg')`). This allows high-resolution multi-dimensional matrices (trends, comparison bars, data spreads, scatter correlations) to render purely in memory, completely eliminating GUI threading crashes on cloud deployment servers.

---

## 🛠️ File Structure & Ecosystem

*   `app.py`: The core user interface framework built on Streamlit, handling cloud inputs and streaming local responses.
*   `data_analyst.py`: The multi-threaded data analytics backend tool running deep statistical workflows on local files.
*   `local_agent.py`: A specialized standalone agent utilizing native tool-binding to solve processing anomalies.
*   `test_chain.py`: A lightweight debugging sandbox testing raw textual completions (`OllamaLLM`).
*   `.gitignore`: Implemented configuration tracking filters to systematically block heavy local dependencies (`.venv/`) and environment keys (`.env`).

---

## 🚀 Execution & Local Deployment

### 1. Clone & Initialize the Workspace
```bash
git clone https://github.com
cd junior-data-analyst-ai-bot-v1
```

### 2. Configure Virtual Environment & Packages
```bash
# Initialize and activate isolated Python environment
python -m venv .venv
.\venv\Scripts\activate  # Windows

# Install dependency matrix
pip install pandas numpy matplotlib seaborn langchain langchain-ollama streamlit
```

### 3. Initialize the Local Inference Brain
Ensure you have [Ollama](https://ollama.com) installed and running locally, then pull the target weight model:
```bash
ollama pull llama3.2
```

### 4. Trigger the Applications
*   **To run the Data Analyst tool with human-in-the-loop security checks:**
    ```bash
    python data_analyst.py
    ```
*   **To deploy the interactive Streamlit user interface browser locally:**
    ```bash
    python -m streamlit run app.py
    ```

---

## 📅 Roadmap: V2 Cloud Architecture Optimization
*   Decouple inference layers completely from local physical hardware.
*   Migrate backend processing from local Ollama loopbacks to low-latency serverless cloud frameworks using free developer API integrations (Groq / Google AI Studio).
*   Enable continuous 24/7 web-browser accessibility with zero background power consumption on target devices.

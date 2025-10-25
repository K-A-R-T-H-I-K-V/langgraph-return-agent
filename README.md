# Orion Labs Support Agent

A stateful, production-ready customer support agent built with LangGraph, Gemini, and Streamlit. This project simulates a real-world support bot for "Orion Labs," a premium electronics company, handling product return requests from start to finish.

---

## Core Features (LLMOps Practices)

This project was built from the ground up to follow MLOps & LLMOps best practices for a reliable, stateful, and production-ready agent.

* **Stateful Memory:** Uses a `langgraph` SQLite checkpointer (`memory.sqlite`) to maintain persistent, multi-turn conversation memory for each user. The agent remembers the conversation context even after a server restart.
* **Separation of Concerns:** The project is strictly modularized:
    * `app.py`: Streamlit frontend (the "View").
    * `agent.py`: LangGraph agent definition and core reasoning (the "Brain").
    * `tools.py`: Deterministic business logic (the "Muscles").
    * `config.py`: Externalized configuration, mock data, and feature flags.
* **Idempotent Tools:** The `initiate_return_ticket` tool is idempotent. It checks the `returns.jsonl` log file before acting, preventing the creation of duplicate return tickets for the same item.
* **Structured Event Logging:** Simulates a real-world microservice architecture by writing a structured JSONL (JSON Lines) entry to `returns.jsonl` for every successful return. This file acts as a persistent, machine-readable "source of truth."
* **Realistic Simulation:**
    * Uses a focused, realistic product catalog (3 core products).
    * Employs human-readable Order IDs (e.g., `ORD-901`) instead of confusing UUIDs.
    * Simulates a full support flow, from lookup to assigning a named pickup agent.
* **Clean Console Logging:** Uses the `rich` library (`logger.py`) to provide clean, color-coded, and readable "thinking" logs in the terminal, showing the agent's ReAct (Reason + Act) loop.
* **Robust Error Handling:** The agent's "manager's script" (`agent.py`) is explicitly instructed on how to handle errors from its tools, such as the "duplicate ticket" error.

---

## Project Structure

Here is the complete file structure for the project, formatted for clarity.

* `/langgraph-return-agent` (Root Directory)
    * `/assets`
        * `logo.png` (Company logo)
        * `user_avatar.png` (Avatar for user messages)
        * `bot_avatar.png` (Avatar for bot messages)
    * `.env` (For storing API keys)
    * `.gitignore`
    * `README.md` (You are here!)
    * `requirements.txt` (All Python dependencies)
    * `agent.py` (LangGraph agent definition & system prompt)
    * `app.py` (Streamlit web frontend)
    * `config.py` (Core settings, model name, & mock databases)
    * `logger.py` (Rich console logging setup)
    * `tools.py` (All business logic tools - `@tool`)
    * `memory.sqlite` (Agent's persistent conversation memory)
    * `returns.jsonl` (Output log of successful returns)

---

## Technical Stack

* **Frontend:** Streamlit
* **Agent Orchestration:** LangGraph
* **LLM:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
* **Stateful Memory:** `langgraph-checkpoint-sqlite`
* **Console Logging:** `rich`
* **Dependencies:** `python-dotenv`, `aiosqlite`

---

## Setup & Run Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/K-A-R-T-H-I-K-V/langgraph-return-agent
    cd langgraph-return-agent
    ```

2.  **Create Conda Environment**
    ```bash
    conda create --name orion-agent python=3.10
    conda activate orion-agent
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set API Key**
    * Create a file named `.env` in the root directory.
    * Add your Google API key to it:
    ```
    GOOGLE_API_KEY="your-google-api-key-here"
    ```

5.  **Add UI Assets**
    * Place your company `logo.png` and your `user_avatar.png` / `bot_avatar.png` inside the `assets/` folder.

6.  **Run the App**
    ```bash
    streamlit run app.py
    ```
    Your browser will open automatically to `http://localhost:8501`.

---

## Example Usage

1.  Open the app and log in using the sidebar. Credentials are in `config.py`:
    * **Username:** `karthik_n`
    * **Password:** `pass123`
2.  Start a conversation, e.g., `"hi i want to return my laptop"`.
3.  The agent will look up your orders and ask you to confirm the item.
4.  It will then check eligibility. If eligible, it will ask for a reason.
5.  Once you provide a reason, the agent will call `initiate_return_ticket`.
6.  You will receive a confirmation with a **Ticket ID** and the name of your **assigned pickup agent**.
7.  After the chat, you can open `returns.jsonl` in your project folder to see the structured JSON log entry for the return you just processed.
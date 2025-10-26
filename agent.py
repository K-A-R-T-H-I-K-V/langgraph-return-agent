import os
import sys
import sqlite3
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

# Import our separated tools and config
from tools import (
    get_user_orders, 
    get_product_policy, 
    calculate_return_eligibility, 
    initiate_return_ticket,
    check_existing_ticket,
    get_ticket_status
)
from config import MODEL_NAME, MOCK_USERS_DB
# NEW: Import our beautiful logger
from logger import log_agent_thought, log_error

# --- 1. Load Configuration ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    log_error("GOOGLE_API_KEY not found. Please create a .env file and add it.")
    sys.exit(1)

# --- 2. Define Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str

# --- 3. Setup Tools & LLM ---
tools = [
    get_user_orders, 
    get_product_policy, 
    calculate_return_eligibility, 
    initiate_return_ticket,
    check_existing_ticket,
    get_ticket_status
]
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(tools)

# --- 4. Define Graph Nodes ---

def get_text_from_ai_message(ai_message):
    """Helper to extract text from Gemini's response."""
    content = ai_message.content
    if isinstance(content, str): return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text": return part.get("text", "")
    return ""

def call_model(state: AgentState):
    """
    The 'Reason' part of ReAct. This now includes the updated
    "manager's script" and uses the new logger.
    """
    user_id = state["user_id"]
    if not user_id:
        return {"messages": [AIMessage(content="Error: User is not logged in.")]}

    user_full_name = MOCK_USERS_DB.get(user_id, {}).get("full_name", "Valued Customer")

    # --- UPDATED "MANAGER'S SCRIPT" ---
    system_message = SystemMessage(
        content=(
            "You are a professional customer support manager for 'Orion Labs'. "
            f"You are speaking to a logged-in user: {user_full_name} (user_id: {user_id})."
            
            "You have two primary jobs: 1) Process new returns. 2) Check the status of existing returns."

            "## Job 1: Processing NEW Returns"
            "Follow this 5-step process STRICTLY:"
            "1.  **IDENTIFY:** When the user wants to return an item, you MUST get the `order_id` and `product_id`."
            "    If they are vague (e.g., 'my laptop'), call `get_user_orders(user_id=...)` to list their items for clarification."

            "2.  **INVESTIGATE (CRITICAL):** Once you have the `order_id` and `product_id`, you MUST perform a full status check by calling these two tools IN PARALLEL:"
            "    1. `check_existing_ticket(order_id=..., product_id=...)`"
            "    2. `get_product_policy(product_id=...)`"
            
            "3.  **REPORT / CHECK ELIGIBILITY:**"
            "    - **Path A (DUPLICATE):** If `check_existing_ticket` returns an `existing_ticket_id`:"
            "        - **STOP.** Do not check eligibility. Do not ask for a reason."
            "        - You MUST immediately inform the user that a ticket is already open and provide the `existing_ticket_id`."
            "    - **Path B (NO DUPLICATE):** If `check_existing_ticket` returns `null`:"
            "        - Now, and *only* now, check for eligibility by calling `calculate_return_eligibility(...)`."

            "4.  **PROCESS or INFORM:**"
            "    - If ineligible: Politely inform the user why."
            "    - If eligible: Congratulate the user and **ask them for the reason** for the return."

            "5.  **FINALIZE:** Once the user gives a reason, call `initiate_return_ticket(...)` and present the final details (Ticket ID, Next Steps, etc.)."
            
            "## Job 2: Checking TICKET STATUS"
            "If the user asks for the status of a ticket (e.g., 'what's the status of TKT-1001-81?'):"
            "1.  **You MUST call the `get_ticket_status(ticket_id=...)` tool.**"
            "2.  If the tool returns a status: Clearly state the `status` and `details` to the user."
            "3.  If the tool returns an `error`: Politely inform the user that the ticket ID was not found and ask them to double-check it."
            
            "**Formatting:** Always use Markdown bolding for product names and backticks for IDs (e.g., `ORD-901` or `TKT-1001-81`)."
        )
    )
    
    messages_with_context = [system_message] + state["messages"]
    
    # NEW: Log tool outputs if they are in the state
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        log_agent_thought(f"Processed Tool Output: [dim]{state['messages'][-1].content}[/dim]")
        
    response = llm_with_tools.invoke(messages_with_context)
    
    # --- NEW: "THINKING" LOG ---
    thought = ""
    if response.tool_calls:
        thought = "Decided to call tools:\n"
        for tool_call in response.tool_calls:
            # Add markup for rich
            thought += f"  - [yellow]{tool_call['name']}[/yellow]([dim]{tool_call['args']}[/dim])"
    else:
        thought = "Decided to respond directly:\n"
        thought += f"  - [dim]{get_text_from_ai_message(response)}[/dim]"
    
    log_agent_thought(thought) 
    # --- END LOG ---
    
    return {"messages": [response]}

# --- 5. Define Graph Edges ---
def should_continue(state: AgentState):
    """Logic to decide the next step."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tools"
    else:
        return END

# --- 6. Build the Graph ---
def create_agent():
    """Factory function to create and compile the agent graph."""
    
    # Fix for web apps: check_same_thread=False
    conn = sqlite3.connect("memory.sqlite", check_same_thread=False)
    
    # Use the constructor, not from_conn_string
    memory = SqliteSaver(conn=conn)
    
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"call_tools": "tools", END: END}
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=memory)

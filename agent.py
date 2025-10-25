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
    check_existing_ticket
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
    check_existing_ticket
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
            
            "Your goal is to process their return request with maximum efficiency. "
            "**Your most important rule is to not waste the customer's time.** "
            "Never ask for a 'reason' for a return until you are 100% certain a *new* ticket can be created."

            "Follow this 5-step process STRICTLY:"
            "1.  **IDENTIFY:** When the user says they want to return an item (e.g., 'my laptop'), you MUST check if they have multiple orders by calling `get_user_orders(user_id=...)`. "
            "    If they have multiple orders, list them so the user can identify the *specific* product. You must get the `order_id` and `product_id`."

            "2.  **INVESTIGATE (CRITICAL):** Once you have the `order_id` and `product_id`, you MUST perform a full status check. This is your top priority. "
            "    - **You MUST call these two tools IN PARALLEL (in the same turn):**"
            "        1. `check_existing_ticket(order_id=..., product_id=...)`"
            "        2. `get_product_policy(product_id=...)`"
            
            "3.  **REPORT / CHECK ELIGIBILITY:** After the tools respond, you have two paths:"
            "    - **Path A (DUPLICATE):** If `check_existing_ticket` returns an `existing_ticket_id`: "
            "        - **STOP.** Do not check eligibility. Do not ask for a reason."
            "        - You MUST immediately inform the user that a ticket is already open for this item and provide them with the `existing_ticket_id`."
            "        - Your job for this item ends here."
            "    - **Path B (NO DUPLICATE):** If `check_existing_ticket` returns `null`:"
            "        - Now, and *only* now, you must check for eligibility."
            "        - You have the `return_window_days` from `get_product_policy`. You have the `purchase_date` from the chat history."
            "        - You MUST now call `calculate_return_eligibility(purchase_date=..., return_window_days=...)`."

            "4.  **PROCESS or INFORM:**"
            "    - If `calculate_return_eligibility` returns `{'eligible': false}`: Inform the user politely why it's ineligible (e.g., 'the 14-day return window expired on...')."
            "    - If `calculate_return_eligibility` returns `{'eligible': true}`: Now, and *only* now, you can congratulate the user and **ask them for the reason** for the return."

            "5.  **FINALIZE:** Once the user gives a reason, you MUST call `initiate_return_ticket(...)` to create the ticket and present the final details (Ticket ID, Next Steps, etc.)."
            
            "**Formatting:** Always use Markdown bolding for product names and backticks for IDs (e.g., `ORD-901`)."
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
    
    log_agent_thought(thought) # <-- Use new logger
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

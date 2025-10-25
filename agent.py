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
    initiate_return_ticket
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
    initiate_return_ticket
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
            
            "Your goal is to process their return request with maximum efficiency and clarity."
            
            "Follow this 5-step process STRICTLY:"
            "1.  **IDENTIFY:** First, understand *which product* the user wants to return. If they are vague (e.g., 'my laptop'), you MUST call `get_user_orders(user_id=...)` to list their purchased items so they can clarify. You need the `order_id` and `product_id`."
            "2.  **CHECK ELIGIBILITY:** Once you have the `order_id` and `product_id`, you MUST check its eligibility. This requires TWO tool calls: `get_product_policy(product_id=...)` AND `calculate_return_eligibility(purchase_date=..., return_window_days=...)`."
            "3.  **INFORM (IF INELIGIBLE):** If `calculate_return_eligibility` returns `{'eligible': false}`, you MUST politely inform the user that the item is not eligible and state the reason (e.g., 'the 14-day return window expired on...'). Your job ends here for this item."
            "4.  **PROCESS (IF ELIGIBLE):** If `calculate_return_eligibility` returns `{'eligible': true}`, you MUST do the following:"
            "    a. Congratulate them on being eligible."
            "    b. You MUST then *immediately* call the `initiate_return_ticket` tool. If you don't have the 'reason' yet, you MUST ask for it."
            "    c. The tool will return a `ticket_id`, `refund_eta`, and `next_steps`."
            "    d. You MUST present ALL this information clearly to the user, including the agent's name mentioned in the `next_steps`."
            "5.  **HANDLE DUPLICATES (CRITICAL):** If the `initiate_return_ticket` tool returns an `{'error': '...', 'existing_ticket_id': '...'}`: "
            "    a. You MUST NOT create a new ticket."
            "    b. You MUST politely inform the user that a return ticket has *already* been created for this item."
            "    c. You MUST provide them with the `existing_ticket_id`."
            
            "Be polite, professional, and clear. **When listing items, use Markdown bolding for product names and backticks for the Order IDs.**"
            "Example: `1. **Orion Laptop 15 Pro** (Order ID: `ORD-901`)`"
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

import json
import random
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool

# Import our realistic mock data and config
from config import (
    MOCK_ORDERS_DB, 
    MOCK_PRODUCTS_DB, 
    MOCK_TICKET_DB, 
    MOCK_RETURN_AGENTS_DB,
    TODAY
)
# Import our beautiful console logger
from logger import log_tool_call, log_error

RETURN_LOG_FILE = "returns.jsonl"

def _read_return_log():
    """Helper function to read the JSONL log file."""
    if not os.path.exists(RETURN_LOG_FILE):
        return []
    with open(RETURN_LOG_FILE, 'r') as f:
        tickets = []
        for line in f:
            try:
                tickets.append(json.loads(line))
            except json.JSONDecodeError:
                continue # Ignore corrupted lines
        return tickets

# --- 1. NEW PROACTIVE TOOL ---
@tool
def check_existing_ticket(order_id: str, product_id: str) -> str:
    """
    Checks if a return ticket already exists for a specific order and product.
    Returns the existing ticket ID if found, otherwise returns null.
    """
    log_tool_call("check_existing_ticket", f"order_id='{order_id}', product_id='{product_id}'")
    all_tickets = _read_return_log()
    for ticket in all_tickets:
        if (ticket.get("order_id") == order_id and
            ticket.get("product_id") == product_id):
            
            log_error(f"Duplicate return found. Ticket: {ticket.get('ticket_id')}")
            # Found one! Return the ticket ID.
            return json.dumps({"existing_ticket_id": ticket.get("ticket_id")})
    
    # No ticket found.
    return json.dumps({"existing_ticket_id": None})

# ... (get_user_orders and get_product_policy are unchanged) ...
@tool
def get_user_orders(user_id: str) -> str:
    """Looks up all orders for a specific user_id."""
    log_tool_call("get_user_orders", f"user_id='{user_id}'")
    user_orders = []
    for order in MOCK_ORDERS_DB:
        if order["user_id"] == user_id:
            product_id = order["product_id"]
            product_details = MOCK_PRODUCTS_DB.get(product_id)
            if product_details:
                enriched_order = order.copy()
                enriched_order["product_name"] = product_details["name"]
                user_orders.append(enriched_order)
    if not user_orders:
        return json.dumps({"error": f"No orders found for user: {user_id}"})
    return json.dumps(user_orders)

@tool
def get_product_policy(product_id: str) -> str:
    """Retrieves the return policy (in days) for a specific product_id (SKU)."""
    log_tool_call("get_product_policy", f"product_id='{product_id}'")
    product = MOCK_PRODUCTS_DB.get(product_id)
    if product is None:
        return json.dumps({"error": f"No product found for ID: {product_id}"})
    return json.dumps({
        "product_id": product_id,
        "product_name": product["name"],
        "return_window_days": product["return_window_days"]
    })

@tool
def calculate_return_eligibility(purchase_date_str: str, return_window_days: int) -> str:
    """Calculates if a product is eligible for return."""
    log_tool_call("calculate_return_eligibility", f"purchase_date='{purchase_date_str}', window={return_window_days}")
    try:
        purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d").date()
        expiry_date = purchase_date + timedelta(days=return_window_days)
        days_left = (expiry_date - TODAY).days
        
        if days_left >= 0:
            return json.dumps({"eligible": True, "expiry_date": expiry_date.isoformat()})
        else:
            return json.dumps({"eligible": False, "expiry_date": expiry_date.isoformat()})
    except Exception as e:
        log_error(f"Error in calculate_return_eligibility: {e}")
        return json.dumps({"error": f"Error calculating eligibility: {str(e)}"})

# --- 2. UPDATED TICKET TOOL (Simpler) ---
@tool
def initiate_return_ticket(order_id: str, product_id: str, reason: str) -> str:
    """
    **CRITICAL TOOL.** Call this *only* after eligibility is confirmed
    AND 'check_existing_ticket' has returned null.
    This tool logs the return and generates a ticket.
    """
    log_tool_call("initiate_return_ticket", f"order_id='{order_id}', product_id='{product_id}', reason='{reason}'")
    
    # We removed the duplicate check because it's now in its own tool.
    
    order_found = any(o["order_id"] == order_id and o["product_id"] == product_id for o in MOCK_ORDERS_DB)
    if not order_found:
        log_error(f"Order/Product ID mismatch. Order: {order_id}, Product: {product_id}")
        return json.dumps({"error": "Order ID or Product ID not found. Cannot initiate return."})

    # Create New Ticket
    MOCK_TICKET_DB["counter"] += 1
    ticket_id = f"TKT-{MOCK_TICKET_DB['counter']}-{random.randint(10, 99)}"
    assigned_agent = random.choice(MOCK_RETURN_AGENTS_DB)
    refund_eta = "5-7 business days after the item is received."
    next_steps = [
        f"1. Please pack the item securely, ideally in its original packaging.",
        f"2. Our return agent, **{assigned_agent['name']}**, will contact you at your registered number to schedule a pickup.",
        f"3. Please have the package ready for them. Your ticket ID is `{ticket_id}`."
    ]
    
    # Create "Beautiful" JSON Log Entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticket_id": ticket_id,
        "order_id": order_id,
        "product_id": product_id,
        "reason": reason,
        "assigned_agent": assigned_agent
    }
    
    # Write to the .jsonl log file
    with open(RETURN_LOG_FILE, 'a') as f:
        json.dump(log_entry, f)
        f.write('\n')
    
    return json.dumps({
        "status": "success",
        "ticket_id": ticket_id,
        "refund_eta": refund_eta,
        "next_steps": next_steps,
    })
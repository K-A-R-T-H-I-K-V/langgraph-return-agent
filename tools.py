import json
import random
import logging
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

# --- MLOps: Set up a dedicated FILE logger for return tickets ---
# This logger will write to `returns.log`
return_log = logging.getLogger('return_tickets')
return_log.setLevel(logging.INFO)

# Check if handlers are already added to avoid duplicates on Streamlit re-runs
if not return_log.hasHandlers():
    file_handler = logging.FileHandler('returns.log')
    file_handler.setLevel(logging.INFO)
    # Format: 2025-10-25 20:30:01 - TICKET CREATED: [Ticket ID: TKT-1001-23]...
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    return_log.addHandler(file_handler)
# --- End of File Logger Setup ---


@tool
def get_user_orders(user_id: str) -> str:
    """
    Looks up all orders for a specific user_id. 
    It enriches the order with the product's common name.
    """
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
    """
    Retrieves the return policy (in days) for a specific product_id (SKU).
    """
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
    """
    Calculates if a product is eligible for return based on its purchase date
    and the return window. Uses the hardcoded 'TODAY' date from config.
    """
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

@tool
def initiate_return_ticket(order_id: str, product_id: str, reason: str) -> str:
    """
    **CRITICAL TOOL.** Call this *only* after eligibility is confirmed.
    This tool creates a log file, assigns a pickup agent, and generates a ticket.
    """
    log_tool_call("initiate_return_ticket", f"order_id='{order_id}', product_id='{product_id}', reason='{reason}'")
    
    order_found = any(o["order_id"] == order_id and o["product_id"] == product_id for o in MOCK_ORDERS_DB)
    if not order_found:
        log_error(f"Order/Product ID mismatch in initiate_return_ticket. Order: {order_id}, Product: {product_id}")
        return json.dumps({"error": "Order ID or Product ID not found. Cannot initiate return."})

    # --- NEW LOGIC ---
    
    # 1. Generate Ticket ID
    MOCK_TICKET_DB["counter"] += 1
    ticket_id = f"TKT-{MOCK_TICKET_DB['counter']}-{random.randint(10, 99)}"
    
    # 2. Assign a simulated return agent
    assigned_agent = random.choice(MOCK_RETURN_AGENTS_DB)
    
    # 3. Create the realistic "Next Steps"
    refund_eta = "5-7 business days after the item is received."
    next_steps = [
        f"1. Please pack the item securely, ideally in its original packaging.",
        f"2. Our return agent, **{assigned_agent['name']}**, will contact you at your registered number to schedule a pickup.",
        f"3. Please have the package ready for them. Your ticket ID is `{ticket_id}`."
    ]
    
    # 4. Write to the returns.log file
    log_message = (
        f"TICKET CREATED: [Ticket ID: {ticket_id}] - "
        f"[Order ID: {order_id}] - [Product: {product_id}] - "
        f"[Reason: {reason}] - "
        f"[Assigned Agent: {assigned_agent['name']} ({assigned_agent['phone']})]"
    )
    return_log.info(log_message)
    
    # 5. Return all info to the LLM
    return json.dumps({
        "status": "success",
        "ticket_id": ticket_id,
        "refund_eta": refund_eta,
        "next_steps": next_steps,
        "assigned_agent": assigned_agent
    })
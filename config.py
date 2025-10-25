from datetime import date
import uuid 

# --- MLOps: Model Configuration ---
MODEL_NAME = "gemini-2.5-flash"

# --- MLOps: Deterministic Test Date ---
# This is our fixed "today" for testing.
TODAY = date(2025, 10, 25)

# --- MLOps: Mock Database (Simulating Reality) ---

# 1. Realistic Product Catalog
# Our company, "Nexora," is famous for 3 products.
# IDs are now realistic SKUs (Stock Keeping Units).
MOCK_PRODUCTS_DB = {
    "NEX-LP-15-PRO": {
        "name": "Nexora Laptop 15 Pro", 
        "return_window_days": 30
    },
    "NEX-PH-7-ULT": {
        "name": "Nexora Phone 7 Ultra", 
        "return_window_days": 14
    },
    "NEX-HP-ELITE-W": {
        "name": "Nexora Elite Headphones (White)", 
        "return_window_days": 30
    },
}

# 2. Realistic User Database
# Keys are usernames. We store a (simple) password and full name.
MOCK_USERS_DB = {
    "karthik_n": {
        "password": "pass123", # In real life, this would be a hash
        "full_name": "Karthik N."
    },
    "sara_j": {
        "password": "pass456",
        "full_name": "Sara Jenkins"
    },
    "david_l": {
        "password": "pass789",
        "full_name": "David Lee"
    }
}

# 3. Realistic Orders Database
# Uses usernames, SKUs, and realistic UUIDs for order IDs.
MOCK_ORDERS_DB = [
    # Karthik bought 2 items
    {
        "order_id": "ORD-901", 
        "user_id": "karthik_n",
        "product_id": "NEX-LP-15-PRO", # Laptop Pro
        "purchase_date": "2025-10-01"
    },
    {
        "order_id": "ORD-902", 
        "user_id": "karthik_n",
        "product_id": "NEX-PH-7-ULT", # Phone 7 Ultra
        "purchase_date": "2025-10-15"
    },
    # Sara bought 2 items
    {
        "order_id": "ORD-903", 
        "user_id": "sara_j",
        "product_id": "NEX-PH-7-ULT",
        "purchase_date": "2025-10-20"
    },
    {
        "order_id": "ORD-904", 
        "user_id": "sara_j",
        "product_id": "NEX-HP-ELITE-W",
        "purchase_date": "2025-08-15" 
    },
    # David bought 1 item
    {
        "order_id": "ORD-905", 
        "user_id": "david_l",
        "product_id": "NEX-LP-15-PRO",
        "purchase_date": "2025-09-20"
    }
]

# 4. Mock Support Ticket System
MOCK_TICKET_DB = {"counter": 1000}

# 5. NEW: Mock Return Agent Database (for realistic simulation)
# These are the "users" who will handle the pickup
MOCK_RETURN_AGENTS_DB = [
    {
        "name": "Alex Chen",
        "phone": "+1 (800) 555-1234"
    },
    {
        "name": "Maria Garcia",
        "phone": "+1 (800) 555-5678"
    },
    {
        "name": "David Kim",
        "phone": "+1 (800) 555-9012"
    }
]
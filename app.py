import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import our agent factory and mock user data
from agent import create_agent
from config import MOCK_USERS_DB
import time

st.set_page_config(page_title="Orion Labs Support", page_icon="assets/logo.png", layout="centered")

# --- MLOps: Caching the Agent ---
@st.cache_resource
def get_agent():
    """Builds and caches the agent graph."""
    print("--- Compiling Agent Graph ---")
    return create_agent()

# --- Helper Function to Extract Text ---
def get_text_from_ai_message(ai_message):
    """Extracts the string content from an AIMessage."""
    content = ai_message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text", "")
    return ""

# --- 1. Initialize Agent & Session State ---
app = get_agent()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Sidebar UI (Login & User Info) ---
st.sidebar.image("assets/logo.png", width=100)

if not st.session_state.user_id:
    st.sidebar.title("Support Login")
    
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            user_data = MOCK_USERS_DB.get(username)
            if user_data and user_data["password"] == password:
                st.session_state.user_id = username
                st.session_state.thread_id = f"{username}_{int(time.time())}"
                st.session_state.messages = [] # Clear messages on new login
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
else:
    # --- Logged-in Sidebar UI ---
    user_name = MOCK_USERS_DB[st.session_state.user_id]["full_name"]
    st.sidebar.title(f"Welcome, {user_name}")
    st.sidebar.caption("You are logged in.")
    
    if st.sidebar.button("Logout"):
        st.session_state.user_id = None
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.rerun()

# --- 3. Main Chat Interface ---

# --- FIX 1: Title with Icon ---
# We use columns to align the icon and title neatly.
col1, col2 = st.columns([1, 6], vertical_alignment="center")
with col1:
    st.image("assets/bot_avatar.png", width=64) # Or 50, adjust as needed
with col2:
    st.title("Orion Labs SupportBot")

# Only print the welcome message if the chat history is empty.
st.write("Welcome to our 24/7 support. I'm here to help with your orders and returns.")

st.divider()

if st.session_state.user_id:
    
    # --- THIS IS THE DEFINITIVE FIX ---

    # 1. Display all messages from history
    # This loop is now the SINGLE source of truth for all messages.
    for message in st.session_state.messages:
        avatar_path = "assets/user_avatar.png" if message["role"] == "user" else "assets/bot_avatar.png"
        with st.chat_message(message["role"], avatar=avatar_path):
            st.markdown(message["content"])

    # 2. Get new user input
    if prompt := st.chat_input("How can I help you with your orders?"):
        # Add user message to state and *immediately* rerun
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # 3. Check if the bot needs to reply
    #    (i.e., if the last message in the history is from the user)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        # 4. Show the "Thinking" spinner
        with st.chat_message("assistant", avatar="assets/bot_avatar.png"):
            with st.spinner("Agent is thinking..."):
                
                # 5. Prepare inputs for the agent
                inputs = {
                    "messages": [HumanMessage(content=st.session_state.messages[-1]["content"])],
                    "user_id": st.session_state.user_id
                }
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # 6. Run the agent
                final_state = app.invoke(inputs, config=config)
                
                ai_response_message = final_state["messages"][-1]
                response_text = get_text_from_ai_message(ai_response_message)
                
                # 7. Add AI message to state (DO NOT PRINT IT)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # 8. Rerun to display the new AI message
                # This forces the script to restart, and the `for` loop (Step 1)
                # will now print the bot's new message.
                st.rerun()

else:
    # --- Show this if not logged in ---
    st.info("Please log in using the sidebar to start a chat.")
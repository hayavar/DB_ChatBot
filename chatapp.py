import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import issuelogs as IPS

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

# Initialize Gemini (via LangChain wrapper)
llm_agent = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True,
    temperature=0.2,
    max_tokens=8192,
    google_api_key=api_key,
)

# Streamlit config
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

# Sidebar menu
with st.sidebar:
    st.title("Navigation")
    menu_selection = st.radio(
        "Go to",
        ["Marketing Assistant", "Issue Debugger"],
    )

# Initialize session state
if "marketing_details_submitted" not in st.session_state:
    st.session_state.marketing_details_submitted = False
if "product_name" not in st.session_state:
    st.session_state.product_name = ""
if "product_purpose" not in st.session_state:
    st.session_state.product_purpose = ""
if "budget_amount" not in st.session_state:
    st.session_state.budget_amount = None
if "timelines" not in st.session_state:
    st.session_state.timelines = ""
if "debugger_messages" not in st.session_state:
    st.session_state.debugger_messages = []
if "debugger_first_visit" not in st.session_state:
    st.session_state.debugger_first_visit = True
if "debugger_user_name" not in st.session_state:
    st.session_state.debugger_user_name = ""
if "waiting_for_issue" not in st.session_state:
    st.session_state.waiting_for_issue = False

# Function to process marketing details by calling crewexecute
def process_marketing_details(product, purpose, budget, timeline):
    st.subheader("Processing Marketing Details...")
    st.write(f"Product Name: {product}")
    st.write(f"Product Purpose: {purpose}")
    st.write(f"Budget: {budget}")
    st.write(f"Timelines: {timeline}")

    from crewassistant import crewexecute
    response_from_crew = crewexecute(product, purpose, budget, timeline)
    st.write("Response from crewexecute:")
    st.write(response_from_crew)

    st.session_state.marketing_details_submitted = True
    st.success("Marketing details submitted and processed!")

# Function to reset the marketing assistant session state
def reset_marketing_assistant():
    st.session_state.marketing_details_submitted = False
    st.session_state.product_name = ""
    st.session_state.product_purpose = ""
    st.session_state.budget_amount = None
    st.session_state.timelines = ""

# Function to display chat messages
def display_chat(messages, key="chat"):
    for i, msg in enumerate(messages):
        message(msg["content"], is_user=(msg["role"] == "user"), key=f"{key}_{i}")

# Placeholder for the getrca() function
def getrca(user_input):
    st.info(f"User input: '{user_input}'")
    with st.spinner("Getting RCA..."):
        try:
            answer_rca = IPS.geetreport(f"My name is {st.session_state.debugger_user_name} and {user_input}")
            st.session_state.debugger_messages.append({"role": "assistant", "content": answer_rca})
            message(answer_rca, is_user=False, key=f"assistant_{len(st.session_state.debugger_messages) - 1}")
        except Exception as e:
            error_message = f"Error in IPS.geetreport: {e}"
            st.session_state.debugger_messages.append({"role": "assistant", "content": error_message})
            message(error_message, is_user=False, key=f"assistant_error")

# Main area based on menu selection
if menu_selection == "Marketing Assistant":
    st.title("📣 Marketing Assistant")

    if not st.session_state.marketing_details_submitted:
        st.subheader("Please provide the following details for your marketing request:")
        product_name = st.text_input("Product Name:", key="product_name_input")
        product_purpose = st.text_area("Product Purpose:", key="product_purpose_input")
        budget_amount = st.number_input("Budget Amount:", min_value=0, key="budget_amount_input")
        timelines = st.text_input("Timelines:", key="timelines_input")

        if st.button("Submit Marketing Details"):
            if product_name and product_purpose and budget_amount is not None and timelines:
                process_marketing_details(product_name, product_purpose, budget_amount, timelines)

            else:
                st.warning("Please fill in all the marketing details.")
    else:
        st.subheader("Marketing Details Submitted")
        st.info("Your marketing details have been submitted and processed.")
        if st.button("Submit Another Request"):
            reset_marketing_assistant()
            st.rerun()

elif menu_selection == "Issue Debugger":
    st.title("🛠️ Issue Debugger")

    if st.session_state.debugger_first_visit:
        st.subheader("Welcome to the Issue Debugger!")
        user_name = st.text_input("Please enter your name:")
        if user_name:
            st.session_state.debugger_user_name = user_name
            st.session_state.debugger_first_visit = False
            st.session_state.waiting_for_issue = True
            st.rerun() # Force a rerun to move to the next stage
    elif st.session_state.waiting_for_issue:
        st.subheader(f"Hello, {st.session_state.debugger_user_name}! Please describe the issue you need help with:")
        if user_input := st.text_area("Enter your issue here:", key="issue_input"):
            getrca(user_input)
            st.session_state.waiting_for_issue = False # Move to the chat display after submitting the issue
            st.rerun() # Force a rerun to show the chat input again
    else:
        st.subheader(f"Hello, {st.session_state.debugger_user_name}! How can I help you with your issue today?")
        display_chat(st.session_state.debugger_messages, key="debugger_chat")
        if user_input := st.chat_input("Type your issue description here..."):
            st.session_state.debugger_messages.append({"role": "user", "content": user_input})
            getrca(user_input)

    if st.sidebar.button("Clear Debugger Chat"):
        st.session_state.debugger_messages = []
        st.session_state.waiting_for_issue = False
        if not st.session_state.debugger_first_visit:
            st.rerun() # Rerun to clear the chat display
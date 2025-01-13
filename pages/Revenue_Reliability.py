import sys
import os
import pandas as pd
import streamlit as st
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_rev_app import TaxAnalyzer

st.set_page_config(page_title="RR Q&A Interface", layout="wide")

st.title("Revenue Reliability Analysis")

# Initialize the TaxAnalyzer
tax_analyzer = TaxAnalyzer()

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Clear chat history functionality
def reset_chat():
    st.session_state.messages = []
    st.rerun()


# Chat interface layout
st.markdown("### 💬 Conversation History")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# Chat input fixed at the bottom of the page
if prompt := st.chat_input("Ask a question about revenue..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate streaming response from TaxAnalyzer
        with st.spinner("Analyzing..."):
            response_chunks = tax_analyzer.chat(prompt)  # Assuming `chat()` yields response chunks
            for chunk in response_chunks:
                full_response += chunk
                message_placeholder.text(full_response + "▌")

        # Finalize response display
        message_placeholder.text(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear button at the bottom right
if st.session_state.messages:
    st.button("Clear ↺", on_click=reset_chat)

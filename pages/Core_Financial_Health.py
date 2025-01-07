import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_app import *
# from asdfgn import *
from datetime import datetime
import pandas as pd

st.title("Core Financial Health Analysis")
# Read from database
df = get_db_data()
df = df[df['tax_period_begin'] != '']
df_x = df.copy()

with st.sidebar:
    # Add helpful information in a clean format
    st.markdown("""
    # CORE FINANCIAL HEALTH
    ## 📚 Guide

    ### 🎯 About This Tool
    Analyzes nonprofit tax records (Form 990) with natural language queries.
    
    **Please enter your EIN to enrich the knowledge base with just your records and 
    for better responses to your queries.**

    ### 💡 Example Questions
    - 🏢 What is the organization's business name?
    - 👥 How many volunteers and employees are there?
    - 💰 What is the total executive compensation?
    - 📈 What are the total contributions?

    ### 📊 Available Information
    - Organization details
    - Financial data
    - Employee counts
    - Compensation data
    - Contributions
    """)

    # Select Ein for Primary Context
    ein_list = df['ein'].unique().tolist()
    ein_list.insert(0, "General Context")
    ein_selected = st.selectbox('*Select EIN*', ein_list)
    if ein_selected != "General Context":
        df = df[df['ein'] == ein_selected]
        if not df.empty:
            business_name = df['business_name'].unique()[0]
            st.success(f"Selected Business: **{business_name}**")

    # Add clear button for chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit Interface
st.title("📊 Nonprofit Tax Record Analysis")

with st.expander("📋 Data Preview"):
    st.dataframe(df.head())

try:
    # Ensure 'total_revenue' column is numeric
    df['total_revenue'] = pd.to_numeric(df['total_revenue'], errors='coerce')

    # Display basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Organizations", df['ein'].nunique())
    with col3:
        avg_revenue = df['total_revenue'].mean()
        st.metric("Avg Revenue", f"${avg_revenue:,.2f}")

    st.markdown("")
    st.markdown("")
    st.markdown("")

    # Create analyzer
    analyzer = TaxAnalyzer()

    # Query input
    query = st.text_input("💭 What would you like to know about the tax records?")
    send_button = st.button("Send")

    if send_button and query:
        with st.spinner("Analyzing..."):
            # Get analysis
            analysis = analyzer.analyze(df, df_x, query, ein_selected)

            # Add to chat history
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append({
                "timestamp": timestamp,
                "query": query,
                "response": analysis
            })

    # Display chat history
    st.markdown("### 💬 Conversation History")
    for chat in reversed(st.session_state.chat_history):
        # Query
        st.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #f0f2f6;">
            <span style="color: #666;">🕒 {chat['timestamp']}</span><br>
            <span style="color: #333;">❓ <b>Question:</b> {chat['query']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Response
        st.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #e8f4ea;">
            <span style="color: #333;">💡 <b>Analysis:</b><br>{chat['response']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px solid #eee;'>",
                    unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error accessing database: {str(e)}")
    st.write("Please ensure the database is properly initialized with tax records.")

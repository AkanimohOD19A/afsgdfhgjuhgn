import streamlit as st

st.set_page_config(
    page_title="Nonprofit Analysis Suite",
    page_icon="📊",
    layout="wide",
)

st.title("Welcome to the Nonprofit Analysis Suite")
st.write("""
This application allows you to analyze nonprofit tax records through two key tools:
- **Core Financial Health Analysis**: Explore core financial data and metrics.
- **Revenue Reliability Analysis**: Dive into trends and stability of revenue sources.

Use the sidebar to navigate between tools.
""")

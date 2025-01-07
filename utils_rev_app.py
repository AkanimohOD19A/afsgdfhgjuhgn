import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from datetime import datetime

load_dotenv()


class RevenueReliabilityAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.conversation_history = []

    def analyze(self, df: pd.DataFrame, query: str) -> str:
        try:
            # Ensure numeric columns and handle NaN
            df = df.fillna(0)

            # Prepare context with the full dataset
            context = "Full Dataset Provided:\n\n"
            context += df.to_string(index=False)

            # Add only the last 2 relevant conversation items
            if self.conversation_history:
                context += "\n\nRecent Conversation Context:\n"
                for q, a in self.conversation_history[-2:]:
                    context += f"\nQ: {q}\nA: {a}\n"

            # System message focused on revenue reliability analysis
            system_message = """You are analyzing nonprofit tax records with a focus on revenue reliability. For each analysis:
                1. Assess consistency and trends in revenue streams (e.g., government grants, fundraising, memberships).
                2. Identify dependencies on single revenue sources and potential risks.
                3. Compare against peer organizations when relevant.
                4. Provide data-backed insights and highlight opportunities to diversify revenue.
                5. Keep responses concise and focused on reliability and sustainability metrics.
                6. Where queries involve predicting or forecasting a value, do not simply return the value of an attribute named 'forecast' or 'predict'. Instead, use the present trend in the dataset to generate a data-driven estimate."""

            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_message,
                messages=[{
                    "role": "user",
                    "content": f"Based on the following tax records:\n\n{context}\n\nQuestion: {query}"
                }],
                max_tokens=1500
            )

            answer = response.content[0].text if response.content else "Unable to generate analysis"
            answer = answer.replace("$ ,", "$ ").replace("  ", " ").replace(" .", ".")

            self.conversation_history.append((query, answer))
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)

            return answer

        except Exception as e:
            return f"Error analyzing records: {str(e)}"

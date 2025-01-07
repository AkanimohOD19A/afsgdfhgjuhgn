import os
import sqlite3
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from anthropic import Anthropic
from datetime import datetime

load_dotenv()


def get_db_data(query=None):
    """Retrieve data from SQLite database with improved error handling"""
    try:
        conn = sqlite3.connect("tax_data.db")
        if query is None:
            query = """
            SELECT *
            FROM tax_form_basic_data
            ORDER BY tax_period_end DESC
            """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()


class TaxAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.conversation_history = []

    def get_summary_stats(self, df, columns_of_interest=None):
        """Get summary statistics for specified columns"""
        if columns_of_interest is None:
            columns_of_interest = [
                'total_revenue', 'total_expenses', 'program_service_expenses',
                'admin_expenses', 'fundraising_expenses', 'total_assets',
                'total_liabilities', 'net_assets'
            ]

        stats = {}
        for col in columns_of_interest:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                }
        return stats

    def analyze(self, df: pd.DataFrame, df_x: pd.DataFrame, query: str, ein_selected: str) -> str:
        try:
            keywords = ["peer", "compare"]
            needs_comparison = any(keyword in query.lower() for keyword in keywords)

            # Start with minimal context
            context = "Analysis Context:\n\n"
            context += f"Dataset Overview:\n"
            context += f"Total Organizations: {df_x['business_name'].nunique()}\n"
            context += f"Date Range: {df_x['tax_period_begin'].min()} to {df_x['tax_period_end'].max()}\n\n"

            if needs_comparison:
                # Get most recent year's data for comparison
                latest_year = df_x['tax_period_end'].max()
                recent_data = df_x[df_x['tax_period_end'] == latest_year]

                # Calculate and include only relevant summary stats
                relevant_columns = [
                    'total_revenue', 'total_expenses', 'program_service_expenses',
                    'admin_expenses', 'fundraising_expenses'
                ]
                stats = self.get_summary_stats(recent_data, relevant_columns)

                context += "Industry Statistics (Most Recent Year):\n"
                for metric, values in stats.items():
                    context += f"\n{metric}:\n"
                    for stat_name, value in values.items():
                        if pd.notnull(value):
                            formatted_value = f"${value:,.2f}" if any(term in metric.lower() for term in
                                                                      ['revenue', 'expenses', 'assets',
                                                                       'liabilities']) else f"{value:,.2f}"
                            context += f"- {stat_name}: {formatted_value}\n"

                # Add selected organization's metrics if available
                if ein_selected != "General Context":
                    selected_data = df[df['tax_period_end'] == latest_year]
                    if not selected_data.empty:
                        context += "\nSelected Organization Metrics:\n"
                        for col in relevant_columns:
                            if col in selected_data.columns:
                                value = selected_data[col].iloc[0]
                                if pd.notnull(value):
                                    formatted_value = f"${value:,.2f}" if any(term in col.lower() for term in
                                                                              ['revenue', 'expenses', 'assets',
                                                                               'liabilities']) else f"{value:,.2f}"
                                    context += f"- {col}: {formatted_value}\n"
            else:
                # For non-comparison queries, only include relevant data for the selected EIN
                if ein_selected != "General Context":
                    selected_df = df[df['ein'] == ein_selected].copy()
                    if not selected_df.empty:
                        # Get most recent record
                        latest_record = selected_df.loc[selected_df['tax_period_end'].idxmax()]
                        context += f"\nMost Recent Data for {latest_record['business_name']}:\n"

                        # Include only the most relevant fields
                        relevant_fields = [
                            'business_name', 'tax_period_end', 'total_revenue',
                            'total_expenses', 'program_service_expenses', 'admin_expenses',
                            'fundraising_expenses', 'total_assets', 'total_liabilities',
                            'net_assets', 'employee_count', 'volunteer_count'
                        ]

                        for field in relevant_fields:
                            if field in latest_record.index and pd.notnull(latest_record[field]):
                                value = latest_record[field]
                                if isinstance(value, (int, float)):
                                    formatted_value = f"${value:,.2f}" if any(term in field.lower() for term in
                                                                              ['revenue', 'expenses', 'assets',
                                                                               'liabilities']) else f"{value:,}"
                                else:
                                    formatted_value = value
                                context += f"- {field}: {formatted_value}\n"

            # Add only the last 2 relevant conversation items
            if self.conversation_history:
                context += "\nRecent Conversation Context:\n"
                for q, a in self.conversation_history[-2:]:
                    context += f"\nQ: {q}\nA: {a}\n"

            # System message focused on efficiency analysis
            system_message = """You are analyzing nonprofit tax records with a focus on program efficiency. For each analysis:
                1. Calculate and compare program efficiency ratios:
                   - Program spending ratio (program expenses / total expenses)
                   - Administrative expense ratio
                   - Fundraising efficiency
                2. Compare against peer organizations when relevant
                3. Provide specific data-backed insights
                4. Suggest potential areas for improvement
                5. Keep responses concise and focused on key metrics"""

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
            answer = answer.replace("$,", "$").replace("  ", " ").replace(" .", ".")

            self.conversation_history.append((query, answer))
            if len(self.conversation_history) > 10:
                self.conversation_history.pop(0)

            return answer

        except Exception as e:
            return f"Error analyzing records: {str(e)}"
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()


class TaxAnalyzer:
    def __init__(self, model="claude-3-sonnet-20240229", max_tokens=1500, temperature=0.7):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # Uses ANTHROPIC_API_KEY from environment
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = []
        self.data_context = {}

    def build_context(self, df):
        """Build comprehensive context with a focus on detailed expense analysis"""
        context_str = "DETAILED EXPENSE ANALYSIS CONTEXT:\n\n"

        # 1. Dataset Overview
        context_str += "1. DATASET OVERVIEW:\n"
        context_str += f"- Total Records: {df.shape[0]}\n"
        context_str += f"- Unique Organizations: {df['BusinessName'].nunique()}\n"
        context_str += f"- Date Range: {df['TaxPeriodBegin'].min()} to {df['TaxPeriodEnd'].max()}\n"

        # 2. Granular Expense Breakdown
        context_str += "\n2. GRANULAR EXPENSE BREAKDOWN:\n"
        expense_categories = ['ProgramServicesAmt', 'ManagementAndGeneralAmt', 'FundraisingAmt']
        for category in expense_categories:
            context_str += f"- Total {category.replace('Amt', '')} Expenses: ${df[category].sum():,.2f}\n"

        # 3. Expense Insights by Group
        context_str += "\n3. EXPENSE INSIGHTS BY GROUP:\n"
        group_expenses = df.groupby('Group').agg({
            'TotalAmt': 'sum',
            'ProgramServicesAmt': 'sum',
            'ManagementAndGeneralAmt': 'sum',
            'FundraisingAmt': 'sum'
        }).sort_values('TotalAmt', ascending=False)
        context_str += group_expenses.to_string() + "\n"

        # 4. Expense Trends Over Time
        context_str += "\n4. EXPENSE TRENDS OVER TIME:\n"
        time_trends = df.groupby(df['TaxPeriodEnd']).agg({ #.dt.year
            'TotalAmt': 'sum',
            'ProgramServicesAmt': 'sum',
            'ManagementAndGeneralAmt': 'sum',
            'FundraisingAmt': 'sum'
        }).round(2)
        context_str += time_trends.to_string() + "\n"

        # 5. Top Organizations by Total Expenses
        context_str += "\n5. TOP ORGANIZATIONS BY TOTAL EXPENSES:\n"
        top_orgs = df.nlargest(10, 'TotalAmt')[['BusinessName', 'TotalAmt', 'ProgramServicesAmt', 'ManagementAndGeneralAmt', 'FundraisingAmt']]
        for _, row in top_orgs.iterrows():
            context_str += f"- {row['BusinessName']}: Total Expenses: ${row['TotalAmt']:,.2f} | "
            context_str += f"Program Services: ${row['ProgramServicesAmt']:,.2f} | "
            context_str += f"Management: ${row['ManagementAndGeneralAmt']:,.2f} | "
            context_str += f"Fundraising: ${row['FundraisingAmt']:,.2f}\n"

        # 6. Efficiency Metrics
        context_str += "\n6. EFFICIENCY METRICS:\n"
        context_str += f"- Program Services Efficiency: {(df['ProgramServicesAmt'].sum() / df['TotalAmt'].sum() * 100):,.2f}%\n"
        context_str += f"- Management and General Efficiency: {(df['ManagementAndGeneralAmt'].sum() / df['TotalAmt'].sum() * 100):,.2f}%\n"
        context_str += f"- Fundraising Efficiency: {(df['FundraisingAmt'].sum() / df['TotalAmt'].sum() * 100):,.2f}%\n"

        # 7. Full Dataset Access
        context_str += "\n7. FULL DATASET ACCESS:\n"
        context_str += "All expense details, organizational metrics, and trends are accessible for analysis.\n"
        context_str += "Available columns for analysis: " + ", ".join(df.columns) + "\n"

        return context_str

    def get_response(self, user_input, df=None):
        """Generate a contextual response to user input"""
        try:
            if df is not None:
                context = self.build_context(df)
            else:
                context = "No data context available."

            # Combine context with conversation history
            full_prompt = (f"{context}\n\nQuestion: {user_input}\n\nProvide detailed but concise and straight forward "
                           f"answer except when told to elaborate using available metrics and historical data.")

            # Create system message
            system_message = """
            You are a financial analyst specialized in the expenses and exposure portion of nonprofit tax records analysis.
            Use the complete dataset to provide comprehensive insights.
            Consider all available metrics, trends, and patterns in your analysis.
            Make connections between different data points to provide deeper insights.
            Support your analysis with specific numbers and trends from the data."""

            # Get response using the Anthropic client
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=[
                    *[{"role": msg["role"], "content": msg["content"]} for msg in self.conversation_history],
                    {"role": "user", "content": full_prompt}
                ]
            )

            # Extract response text
            assistant_reply = response.content[0].text

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": assistant_reply})

            # Maintain reasonable history length
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return assistant_reply

        except Exception as e:
            print(f"Error in get_response: {str(e)}")  # Add debugging print
            return f"Error generating response: {str(e)}"

    def chat(self, query):
        """Main interface for chatting with the bot"""
        try:
            # Get and prepare data
            df = pd.read_csv("tax_filing_revenue.csv")

            # Generate response
            return self.get_response(query, df)

        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return f"Error processing request: {str(e)}"

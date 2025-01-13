import os
import re
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

    def prep_data(self, df):
        """Prepare and clean data for analysis."""
        # Ensure column names are uniform
        df.columns = df.columns.str.strip().str.lower()

        # Filter out rows with null or empty `tax_period_end`
        df = df[df['tax_period_end'].notnull() & (df['tax_period_end'] != '')]
        df = df.dropna(subset=['business_name'])

        # Convert specified columns to lowercase
        cols_to_lower = ['business_name', 'website']
        for col in cols_to_lower:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()

        # Replace invalid strings and handle NaN values
        df = df.replace(['', 'NA', 'false', 'true'], [None, None, False, True])

        # Convert numeric columns
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert date columns
        date_cols = ['tax_period_begin', 'tax_period_end']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        if 'formation_year' in df.columns:
            df['formation_year'] = pd.to_datetime(df['formation_year'], format='%Y', errors='coerce')

        # Identify dynamic program service columns
        program_service_cols = [col for col in df.columns if
                                re.match(r'program_service_\d+_totalrevenuecolumnamt', col, re.IGNORECASE)]
        program_service_desc_cols = [col for col in df.columns if
                                     re.match(r'program_service_\d+_desc', col, re.IGNORECASE)]

        # Sum up total program service revenue
        if program_service_cols:
            df['total_program_service_revenue'] = df[program_service_cols].sum(axis=1)
        else:
            df['total_program_service_revenue'] = 0

        # Create a combined description column for program services
        if program_service_desc_cols:
            df['program_service_descriptions'] = df[program_service_desc_cols].astype(str).apply(
                lambda x: ', '.join(x.dropna()), axis=1
            )
        else:
            df['program_service_descriptions'] = ''

        return df

    def build_context(self, df):
        """Build comprehensive context for analysis."""
        context_str = "COMPREHENSIVE DATASET ANALYSIS:\n\n"

        # 1. Dataset Overview
        context_str += "1. DATASET OVERVIEW:\n"
        context_str += f"- Total Records: {df.shape[0]}\n"
        context_str += f"- Unique Organizations: {df['business_name'].nunique()}\n"
        context_str += f"- Date Range: {df['tax_period_end'].min()} to {df['tax_period_end'].max()}\n"

        # 2. Financial Overview
        context_str += "\n2. FINANCIAL METRICS:\n"
        context_str += f"- Total Revenue Range: ${df['total_revenue'].min():,.2f} to ${df['total_revenue'].max():,.2f}\n"
        context_str += f"- Average Revenue: ${df['total_revenue'].mean():,.2f}\n"
        # context_str += f"- Average Net Income: ${(df['total_revenue'] - df['total_expenses']).mean():,.2f}\n"

        # 3. Program Service Metrics
        context_str += "\n3. PROGRAM SERVICE METRICS:\n"
        context_str += f"- Total Program Service Revenue: ${df['total_program_service_revenue'].sum():,.2f}\n"
        context_str += f"- Average Revenue per Program Service: ${(df['total_program_service_revenue'].sum() / len(df)):,.2f}\n"
        context_str += "- Program Services Descriptions:\n"
        for desc in df['program_service_descriptions'].unique():
            context_str += f"  - {desc}\n"

        # 4. Revenue Trends
        context_str += "\n4. REVENUE TRENDS BY YEAR:\n"
        revenue_trends = df.groupby(df['tax_period_end'].dt.year)['total_revenue'].agg(['count', 'mean', 'sum']).round(2)
        context_str += revenue_trends.to_string() + "\n"

        # 5. Top Organizations by Revenue
        context_str += "\n5. TOP ORGANIZATIONS BY REVENUE:\n"
        top_orgs = df.nlargest(10, 'total_revenue')[['business_name', 'total_revenue', 'total_program_service_revenue']]
        for _, row in top_orgs.iterrows():
            context_str += f"- {row['business_name']}: Total Revenue: ${row['total_revenue']:,.2f}, Program Service Revenue: ${row['total_program_service_revenue']:,.2f}\n"

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
            You are a financial analyst specialized in the inwards/revenue of nonprofit tax records analysis.
            Use the complete dataset to provide comprehensive insights.
            Consider all available metrics, trends, and patterns in your analysis.
            Make connections between different data points to provide deeper insights.
            Support your analysis with specific numbers and trends from the data
            Keep to a concise and straight forward answer except when told to elaborate"""

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
            cleaned_df = self.prep_data(df)

            # Generate response
            return self.get_response(query, cleaned_df)

        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return f"Error processing request: {str(e)}"

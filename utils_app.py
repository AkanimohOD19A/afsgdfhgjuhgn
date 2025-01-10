import os
import requests
import sqlite3
import pandas as pd
from datetime import datetime
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
        self.col_order = [
            'ein', 'business_name', 'formation_year', 'tax_period_begin',
            'tax_period_end', 'total_revenue', 'total_expenses', 'net_income',
            'net_assets_boy', 'net_assets_eoy', 'total_contributions',
            'program_service_revenue', 'program_services_expenses',
            'bond_proceeds', 'royalties', 'rental_property_income',
            'net_fundraising', 'sales_of_assets', 'net_inventory_sales',
            'investment_income', 'other_revenue', 'unrelated_business_revenue',
            'related_or_exempt_func_income', 'operating_margin',
            'executive_compensation', 'professional_fundraising_fees',
            'other_salaries_and_wages', 'other_compensations',
            'total_employees', 'total_volunteers', 'fundraising_expenses',
            'management_and_general_expenses', 'total_assets_boy',
            'total_assets_eoy', 'total_liabilities_boy', 'total_liabilities_eoy',
            'compensation_from_other_srcs', 'contrct_rcvd_greater_than_100k',
            'total_comp_greater_than_150k', 'indiv_rcvd_greater_than_100k',
            'tot_reportable_comp_rltd_org', 'program_efficiency',
            'exclusion_amount', 'group_return_for_affiliates',
            'total_gross_ubi', 'voting_members_governing_body',
            'voting_members_independent', 'website'
        ]

    def prep_data(self, df):
        # Remove rows with empty 'tax_period_end'
        df = df[df['tax_period_end'].notnull() & (df['tax_period_end'] != '')]

        # Drop rows with missing 'business_name'
        df = df.dropna(subset=['business_name'])

        # Convert specific columns to lowercase
        cols_to_lower = ['business_name', 'website']
        df[cols_to_lower] = df[cols_to_lower].apply(lambda x: x.str.lower())

        # Replace invalid strings and NaN with zeros
        df = df.replace(['', 'NA', 'false', 'true'], [None, None, False, True])

        numeric_cols = [
            'total_revenue', 'net_assets_eoy', 'net_assets_boy',
            'total_expenses', 'total_contributions', 'program_service_revenue',
            'total_comp_greater_than_150k', 'compensation_from_other_srcs'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert specific columns to boolean-like values
        bool_like_cols = ['group_return_for_affiliates', 'compensation_from_other_srcs', 'total_comp_greater_than_150k']
        for col in bool_like_cols:
            df[col] = df[col].astype(bool)

        # Convert date columns
        date_cols = ['tax_period_begin', 'tax_period_end']
        df[date_cols] = df[date_cols].apply(pd.to_datetime, errors='coerce')
        df['formation_year'] = pd.to_datetime(df['formation_year'], format='%Y', errors='coerce')

        # Calculate derived columns
        df['net_income'] = df['total_revenue'] - df['net_assets_eoy']

        # Reorder columns
        cleaned_dataframe = df[self.col_order]

        return cleaned_dataframe

    def build_context(self, df):
        """Build comprehensive context for the analysis"""
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
        context_str += f"- Total Assets Range: ${df['total_assets_eoy'].min():,.2f} to ${df['total_assets_eoy'].max():,.2f}\n"
        context_str += f"- Average Net Income: ${df['net_income'].mean():,.2f}\n"

        # 3. Organizational Metrics
        context_str += "\n3. ORGANIZATIONAL METRICS:\n"
        context_str += f"- Average Employees: {df['total_employees'].mean():,.0f}\n"
        context_str += f"- Average Volunteers: {df['total_volunteers'].mean():,.0f}\n"
        context_str += f"- Organizations with Websites: {df['website'].notnull().sum()}\n"

        # 4. Revenue Trends
        context_str += "\n4. REVENUE TRENDS BY YEAR:\n"
        revenue_trends = df.groupby(df['tax_period_end'].dt.year).agg({
            'total_revenue': ['count', 'mean', 'sum'],
            'operating_margin': 'mean'
        }).round(2)
        context_str += revenue_trends.to_string() + "\n"

        # 5. Top Organizations
        context_str += "\n5. TOP ORGANIZATIONS BY REVENUE:\n"
        top_orgs = df.nlargest(10, 'total_revenue')[['business_name', 'total_revenue', 'total_employees', 'operating_margin']]
        for _, row in top_orgs.iterrows():
            context_str += f"- {row['business_name']}: ${row['total_revenue']:,.2f} | "
            context_str += f"Employees: {row['total_employees']:,.0f} | "
            context_str += f"Margin: {row['operating_margin']}%\n"

        # 6. Program Efficiency
        context_str += "\n6. PROGRAM EFFICIENCY METRICS:\n"
        context_str += f"- Average Program Efficiency: {df['program_efficiency'].mean():,.2f}%\n"
        context_str += f"- Program Services vs Total Expenses: {(df['program_services_expenses'].sum() / df['total_expenses'].sum() * 100):,.2f}%\n"

        # 7. Compensation Insights
        context_str += "\n7. COMPENSATION INSIGHTS:\n"
        context_str += f"- Average Executive Compensation: ${df['executive_compensation'].mean():,.2f}\n"
        context_str += f"- Organizations with High Compensation (>150k): {df['total_comp_greater_than_150k'].sum()}\n"

        # 8. Full Dataset Access
        context_str += "\n8. FULL DATASET ACCESS:\n"
        context_str += "All financial metrics, organizational data, and historical trends are available for analysis.\n"
        context_str += "Available columns for analysis: " + ", ".join(self.col_order) + "\n"

        return context_str

    def get_response(self, user_input, df=None):
        """Generate a contextual response to user input"""
        try:
            if df is not None:
                context = self.build_context(df)
            else:
                context = "No data context available."

            # Combine context with conversation history
            full_prompt = f"{context}\n\nQuestion: {user_input}\n\nProvide detailed analysis using all available metrics and historical data."

            # Create system message
            system_message = """You are a financial analyst specialized in nonprofit tax records analysis.
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
            df = pd.read_csv("tax_filing_glimpse.csv")
            cleaned_df = self.prep_data(df)

            # Generate response
            return self.get_response(query, cleaned_df)

        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return f"Error processing request: {str(e)}"

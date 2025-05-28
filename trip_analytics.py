import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
from supabase_fetch import supabase
from llm_interaction import get_llm_insights

class TripAnalytics:
    def __init__(self):
        self.supabase = supabase
        
    def fetch_trip_data(self, trip_name=None):
        """Fetch trip data from Supabase"""
        query = self.supabase.table('trips').select('*')
        if trip_name:
            query = query.eq('name', trip_name)
        response = query.execute()
        return pd.DataFrame(response.data)
    
    def fetch_expenses(self, trip_name=None):
        """Fetch expense data from Supabase"""
        try:
            if trip_name:
                # First get the trip ID from the name
                trip_data = self.fetch_trip_data(trip_name)
                if trip_data.empty:
                    return pd.DataFrame()
                trip_id = trip_data.iloc[0]['id']
                query = self.supabase.table('expenses').select('*').eq('trip_id', trip_id)
            else:
                query = self.supabase.table('expenses').select('*')
            response = query.execute()
            df = pd.DataFrame(response.data)
            
            # Ensure required columns exist
            required_columns = ['amount', 'category', 'created_at']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in expenses data: {missing_columns}")
                return pd.DataFrame()
                
            # Convert created_at to datetime and create date column
            if 'created_at' in df.columns:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
            else:
                print("Warning: created_at column not found in expenses data")
                return pd.DataFrame()
                
            return df
        except Exception as e:
            print(f"Error fetching expenses: {e}")
            return pd.DataFrame()
    
    def generate_expense_distribution(self, trip_name=None):
        """Generate expense distribution visualization"""
        expenses_df = self.fetch_expenses(trip_name)
        if expenses_df.empty:
            return None
            
        # Group by category and sum amounts
        category_totals = expenses_df.groupby('category')['amount'].sum().reset_index()
        
        fig = px.pie(category_totals, 
                    values='amount', 
                    names='category',
                    title=f'Expense Distribution by Category{f" - {trip_name}" if trip_name else ""}',
                    hole=0.4)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def generate_trend_analysis(self, trip_name=None):
        """Generate expense trends over time"""
        expenses_df = self.fetch_expenses(trip_name)
        if expenses_df.empty:
            return None
            
        try:
            # Group by date and sum amounts
            daily_expenses = expenses_df.groupby('date')['amount'].sum().reset_index()
            
            # Convert date to string for plotting
            daily_expenses['date'] = daily_expenses['date'].astype(str)
            
            fig = px.line(daily_expenses, 
                         x='date', 
                         y='amount',
                         title=f'Daily Expense Trends{f" - {trip_name}" if trip_name else ""}',
                         labels={'amount': 'Total Expenses', 'date': 'Date'})
            
            fig.update_layout(xaxis_title='Date', yaxis_title='Amount')
            return fig
        except Exception as e:
            print(f"Error generating trend analysis: {e}")
            return None
    
    def generate_budget_comparison(self, trip_name):
        """Compare actual expenses with budget"""
        trip_data = self.fetch_trip_data(trip_name)
        expenses_df = self.fetch_expenses(trip_name)
        
        if trip_data.empty or expenses_df.empty:
            return None
            
        budget = trip_data.iloc[0]['budget']
        total_expenses = expenses_df['amount'].sum()
        
        fig = go.Figure(data=[
            go.Bar(name='Budget', x=['Budget'], y=[budget], marker_color='blue'),
            go.Bar(name='Actual', x=['Actual'], y=[total_expenses], marker_color='red')
        ])
        
        fig.update_layout(
            title=f'Budget vs Actual Expenses - {trip_name}',
            barmode='group',
            yaxis_title='Amount'
        )
        return fig
    
    def generate_ai_insights(self, trip_name):
        """Generate AI-powered insights using LLM"""
        expenses_df = self.fetch_expenses(trip_name)
        trip_data = self.fetch_trip_data(trip_name)
        
        if expenses_df.empty or trip_data.empty:
            return None
            
        # Prepare data for LLM analysis
        total_expenses = float(expenses_df['amount'].sum())  # Convert to Python float
        budget = float(trip_data.iloc[0]['budget'])  # Convert to Python float
        expense_categories = {str(k): float(v) for k, v in expenses_df.groupby('category')['amount'].sum().to_dict().items()}  # Convert keys to str and values to float
        trip_duration = int((pd.to_datetime(trip_data.iloc[0]['end_date']) - 
                           pd.to_datetime(trip_data.iloc[0]['start_date'])).days)  # Convert to Python int
        average_daily_expense = float(total_expenses / trip_duration)  # Convert to Python float
        
        analysis_data = {
            'trip_name': trip_name,
            'total_expenses': total_expenses,
            'budget': budget,
            'expense_categories': expense_categories,
            'trip_duration': trip_duration,
            'average_daily_expense': average_daily_expense
        }
        
        # Get insights from LLM
        prompt = f"""Analyze the following trip expense data and provide insights:
        {json.dumps(analysis_data, indent=2)}
        
        Please provide:
        1. Key spending patterns
        2. Budget compliance analysis
        3. Recommendations for future trips
        4. Anomaly detection in expenses
        """
        
        insights = get_llm_insights(prompt)
        return insights
    
    def generate_expense_clusters(self, trip_name=None):
        """Generate expense clusters using K-means"""
        expenses_df = self.fetch_expenses(trip_name)
        if expenses_df.empty:
            return None
            
        try:
            # Prepare data for clustering
            X = expenses_df[['amount']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            expenses_df['cluster'] = kmeans.fit_predict(X_scaled)
            
            # Convert date to string for plotting
            expenses_df['date'] = expenses_df['date'].astype(str)
            
            # Create visualization
            fig = px.scatter(expenses_df, 
                            x='date', 
                            y='amount',
                            color='cluster',
                            title=f'Expense Clusters{f" - {trip_name}" if trip_name else ""}',
                            labels={'amount': 'Amount', 'date': 'Date'})
            
            return fig
        except Exception as e:
            print(f"Error generating expense clusters: {e}")
            return None
    
    def get_all_analytics(self, trip_name=None):
        """Generate all analytics for a trip"""
        return {
            'expense_distribution': self.generate_expense_distribution(trip_name),
            'trend_analysis': self.generate_trend_analysis(trip_name),
            'budget_comparison': self.generate_budget_comparison(trip_name) if trip_name else None,
            'expense_clusters': self.generate_expense_clusters(trip_name),
            'ai_insights': self.generate_ai_insights(trip_name) if trip_name else None
        } 
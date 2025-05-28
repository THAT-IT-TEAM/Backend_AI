import unittest
import json
from waitress_server import app
import pandas as pd
from unittest.mock import patch, MagicMock
from trip_analytics import TripAnalytics
from datetime import datetime

class TestAnalyticsEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
        # Create dates for the sample data
        dates = pd.date_range(start='2024-01-01', periods=5)
        
        # Sample data for mocking
        self.sample_expenses = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500],
            'category': ['Food', 'Transport', 'Hotel', 'Food', 'Transport'],
            'created_at': dates,
            'date': dates,  # Add date column for plotting
            'trip_id': ['123'] * 5
        })
        
        # Convert dates to strings for JSON serialization
        self.sample_trip = pd.DataFrame({
            'id': ['123'],
            'name': ['Test Trip'],
            'budget': [2000],
            'start_date': ['2024-01-01'],
            'end_date': ['2024-01-05']
        })

    @patch('trip_analytics.TripAnalytics.fetch_expenses')
    @patch('trip_analytics.TripAnalytics.fetch_trip_data')
    def test_get_all_analytics(self, mock_fetch_trip, mock_fetch_expenses):
        """Test the /api/analytics/all endpoint"""
        # Mock the data fetching
        mock_fetch_expenses.return_value = self.sample_expenses.copy()
        mock_fetch_trip.return_value = self.sample_trip.copy()
        
        # Make the request
        response = self.app.get('/api/analytics/all')
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify the response contains all expected analytics
        self.assertIn('expense_distribution', data)
        self.assertIn('trend_analysis', data)
        self.assertIn('expense_clusters', data)
        
        # Verify the plots are returned as JSON strings
        self.assertIsInstance(data['expense_distribution'], str)
        self.assertIsInstance(data['trend_analysis'], str)
        self.assertIsInstance(data['expense_clusters'], str)

    @patch('trip_analytics.get_llm_insights')  # Patch at the import location in trip_analytics
    @patch('trip_analytics.TripAnalytics.fetch_expenses')
    @patch('trip_analytics.TripAnalytics.fetch_trip_data')
    def test_get_trip_analytics(self, mock_fetch_trip, mock_fetch_expenses, mock_llm_insights):
        """Test the /api/analytics/trip endpoint with trip name"""
        # Mock the data fetching
        mock_fetch_expenses.return_value = self.sample_expenses.copy()
        mock_fetch_trip.return_value = self.sample_trip.copy()
        mock_llm_insights.return_value = "Sample AI insights"  # Mock LLM response
        
        # Test data
        test_data = {
            'trip_name': 'Test Trip'
        }
        
        # Make the request
        response = self.app.post(
            '/api/analytics/trip',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify the response contains all expected analytics
        self.assertIn('expense_distribution', data)
        self.assertIn('trend_analysis', data)
        self.assertIn('budget_comparison', data)
        self.assertIn('expense_clusters', data)
        self.assertIn('ai_insights', data)
        
        # Verify the plots are returned as JSON strings
        self.assertIsInstance(data['expense_distribution'], str)
        self.assertIsInstance(data['trend_analysis'], str)
        self.assertIsInstance(data['budget_comparison'], str)
        self.assertIsInstance(data['expense_clusters'], str)
        self.assertIsInstance(data['ai_insights'], str)

    def test_get_trip_analytics_missing_name(self):
        """Test the /api/analytics/trip endpoint with missing trip name"""
        # Test data without trip_name
        test_data = {}
        
        # Make the request
        response = self.app.post(
            '/api/analytics/trip',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Missing trip_name in request body')

    @patch('trip_analytics.TripAnalytics.fetch_expenses')
    @patch('trip_analytics.TripAnalytics.fetch_trip_data')
    def test_get_trip_analytics_no_data(self, mock_fetch_trip, mock_fetch_expenses):
        """Test the /api/analytics/trip endpoint with non-existent trip"""
        # Mock empty data
        mock_fetch_expenses.return_value = pd.DataFrame()
        mock_fetch_trip.return_value = pd.DataFrame()
        
        # Test data
        test_data = {
            'trip_name': 'Non Existent Trip'
        }
        
        # Make the request
        response = self.app.post(
            '/api/analytics/trip',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify all analytics are None when no data is found
        self.assertIsNone(data['expense_distribution'])
        self.assertIsNone(data['trend_analysis'])
        self.assertIsNone(data['budget_comparison'])
        self.assertIsNone(data['expense_clusters'])
        self.assertIsNone(data['ai_insights'])

    @patch('trip_analytics.TripAnalytics.fetch_expenses')
    def test_get_all_analytics_no_data(self, mock_fetch_expenses):
        """Test the /api/analytics/all endpoint with no data"""
        # Mock empty data
        mock_fetch_expenses.return_value = pd.DataFrame()
        
        # Make the request
        response = self.app.get('/api/analytics/all')
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify all analytics are None when no data is found
        self.assertIsNone(data['expense_distribution'])
        self.assertIsNone(data['trend_analysis'])
        self.assertIsNone(data['expense_clusters'])

if __name__ == '__main__':
    unittest.main() 
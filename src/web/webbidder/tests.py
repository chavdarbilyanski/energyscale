# src/web/webbidder/tests.py

import os
from io import StringIO
from unittest.mock import patch, MagicMock

from django.test import TestCase, override_settings
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from .forms import CSVUploadForm
from .services import run_rl_model_simulation

# --- Test Suite for Views and Forms ---

class WebbidderViewAndFormTests(TestCase):
    """Tests the web-facing parts of the application: views and forms."""

    def test_upload_page_get_request(self):
        """
        Tests that the main upload page loads correctly on a GET request.
        This is a fundamental "smoke test".
        """
        url = reverse('upload_csv')
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'upload.html')
        self.assertIsInstance(response.context['form'], CSVUploadForm)
        self.assertContains(response, "Upload CSV File")

    def test_form_is_valid_with_correct_data(self):
        """
        Tests that the CSVUploadForm is valid when given correct data.
        """
        # Create a dummy in-memory CSV file for the form
        csv_data = "Date;Hour;Price (EUR)\n01/01/2023;1;100.0"
        csv_file = SimpleUploadedFile("test.csv", csv_data.encode('utf-8'), content_type="text/csv")
        
        form_data = {
            'max_battery_capacity': 100.0,
            'charge_discharge_rate': 50.0,
        }
        
        form = CSVUploadForm(form_data, {'csv_file': csv_file})
        self.assertTrue(form.is_valid())

    def test_form_is_invalid_with_negative_capacity(self):
        """
        Tests that the form's validators correctly reject negative numbers.
        """
        csv_data = "Date;Hour;Price (EUR)\n01/01/2023;1;100.0"
        csv_file = SimpleUploadedFile("test.csv", csv_data.encode('utf-8'), content_type="text/csv")

        form_data = {
            'max_battery_capacity': -100.0, # Invalid value
            'charge_discharge_rate': 50.0,
        }

        form = CSVUploadForm(form_data, {'csv_file': csv_file})
        self.assertFalse(form.is_valid())
        self.assertIn('max_battery_capacity', form.errors)
        self.assertIn('must be a positive number', form.errors['max_battery_capacity'][0])


# --- Test Suite for the Service Layer ---

# Create a temporary directory for our fake model files during testing
TEST_MODEL_DIR = os.path.join(settings.BASE_DIR, 'test_models')

@override_settings(BASE_DIR=TEST_MODEL_DIR)
class ServiceLayerTests(TestCase):
    """
    Tests the core business logic in services.py.
    This test is isolated from the web layer.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create fake model and stats files before any tests run.
        This runs only once for the entire test class.
        """
        super().setUpClass()
        os.makedirs(TEST_MODEL_DIR, exist_ok=True)
        # We can just create empty files. The service logic only checks for existence.
        # For a more advanced test, you'd use real (but small) test models.
        with open(os.path.join(TEST_MODEL_DIR, 'battery_ppo_agent_v2.zip'), 'w') as f:
            f.write("dummy_model_data")
        with open(os.path.join(TEST_MODEL_DIR, 'vec_normalize_stats_v2.pkl'), 'w') as f:
            f.write("dummy_stats_data")

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the fake model files after all tests have run.
        """
        super().tearDownClass()
        for f in os.listdir(TEST_MODEL_DIR):
            os.remove(os.path.join(TEST_MODEL_DIR, f))
        os.rmdir(TEST_MODEL_DIR)

    @patch('webbidder.services.PPO.load')
    @patch('webbidder.services.VecNormalize.load')
    def test_run_rl_model_simulation_with_mocked_model(self, mock_vec_load, mock_ppo_load):
        """
        Tests the end-to-end data processing logic.
        We "mock" the actual ML model loading to speed up the test and isolate the logic.
        """
        # --- Arrange (Set up the test) ---
        
        # Configure the mock objects to return a predictable "model"
        mock_model = MagicMock()
        # Make our mock model always predict action "1" (BUY)
        mock_model.predict.return_value = ([1], None)
        mock_ppo_load.return_value = mock_model
        
        mock_vec_env = MagicMock()
        # Make our mock normalizer return a dummy array
        mock_vec_env.normalize_obs.return_value = [[0.5, 0.5]] 
        mock_vec_load.return_value = mock_vec_env
        
        # Create a sample CSV in memory
        csv_data = "Date;Hour;Price (EUR)\n01/01/2023;1;100.0\n01/01/2023;2;200.0"
        csv_file = StringIO(csv_data) # Use StringIO to simulate a file

        # --- Act (Run the function we want to test) ---
        battery_charge, total_profit, results = run_rl_model_simulation(
            csv_file=csv_file,
            max_battery_capacity=100,
            charge_discharge_rate=50
        )

        # --- Assert (Check if the results are what we expect) ---
        self.assertEqual(len(results), 2) # Should have processed two rows
        
        # Check the first action (it should be a BUY)
        first_result = results[0]
        self.assertEqual(first_result['action'], 'BUY')
        self.assertEqual(first_result['price'], 0.1) # 100 EUR/MWh = 0.1 EUR/kWh
        self.assertAlmostEqual(first_result['profit_change'], -5.0) # -0.1 EUR/kWh * 50 kWh = -5.0

        # Check the final state
        self.assertAlmostEqual(battery_charge, 100) # 0 + 50 (buy) + 50 (buy)
        self.assertAlmostEqual(total_profit, -30.0) # -5.0 (first hour) + (-0.2 * 50) = -5 - 10 = -15. Wait...
        # Let's re-calculate profit based on the code:
        # Hour 1: battery=0, price=0.1, buy 50. profit_change = -5. total_profit=-5. new_battery=50.
        # Hour 2: battery=50, price=0.2, buy 50. profit_change = -10. total_profit=-15. new_battery=100.
        self.assertAlmostEqual(total_profit, -15.0)
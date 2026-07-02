import sys
import types
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WEB_APP_DIR = ROOT / "web_app"
sys.path.insert(0, str(WEB_APP_DIR))


streamlit_stub = types.SimpleNamespace(cache_resource=lambda func: func)
sys.modules.setdefault("streamlit", streamlit_stub)

import utils  # noqa: E402


class DummyModel:
    def predict(self, frame):
        return frame["LivingArea"].to_numpy() * 2


class WebAppUtilsTest(unittest.TestCase):
    def make_valid_frame(self):
        return pd.DataFrame([{feature: 1 for feature in utils.get_required_features()}])

    def test_validate_uploaded_data_rejects_empty_frame(self):
        is_valid, message = utils.validate_uploaded_data(pd.DataFrame())

        self.assertFalse(is_valid)
        self.assertEqual(message, "Uploaded file is empty.")

    def test_validate_uploaded_data_reports_missing_features(self):
        is_valid, message = utils.validate_uploaded_data(pd.DataFrame({"LivingArea": [1800]}))

        self.assertFalse(is_valid)
        self.assertIn("Missing", message)
        self.assertIn("required column", message)

    def test_prepare_features_orders_columns_for_model(self):
        frame = self.make_valid_frame()
        reversed_frame = frame[list(reversed(utils.get_required_features()))]

        prepared = utils.prepare_features(reversed_frame)

        self.assertEqual(list(prepared.columns), utils.get_required_features())

    def test_predict_from_dataframe_uses_prepared_features(self):
        frame = self.make_valid_frame()
        frame["LivingArea"] = 10

        predictions = utils.predict_from_dataframe(DummyModel(), frame)

        self.assertEqual(predictions.tolist(), [20])


if __name__ == "__main__":
    unittest.main()

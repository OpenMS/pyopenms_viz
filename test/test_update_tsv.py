import os
import unittest
import pandas as pd
from update_tsv_sync import extract_parameters_from_conf, update_tsv_file, TSV_DIR, TSV_MAPPING

class TestUpdateTSV(unittest.TestCase):

    def setUp(self):
        """Setup temporary TSV files before each test."""
        self.sample_conf = {
            "chromatogram_param": 10,
            "mobilogram_param": 20,
            "spectrum_param": "TestValue",
            "unknown_param": 50  # Should go to baseplot.tsv
        }

        # Create test TSV files
        os.makedirs(TSV_DIR, exist_ok=True)
        for file_name in TSV_MAPPING.values():
            file_path = os.path.join(TSV_DIR, file_name)
            pd.DataFrame(columns=["Parameter", "Default", "Type", "Description"]).to_csv(file_path, sep="\t", index=False)

    def test_update_tsv_file(self):
        """Test if conf.py changes update the correct TSV files."""
        update_tsv_file(self.sample_conf)

        for category, file_name in TSV_MAPPING.items():
            file_path = os.path.join(TSV_DIR, file_name)
            df = pd.read_csv(file_path, sep="\t")

            if category == "baseplot":
                self.assertIn("unknown_param", df["Parameter"].values)
            else:
                expected_param = f"{category}_param"
                self.assertIn(expected_param, df["Parameter"].values)

    def tearDown(self):
        """Clean up test TSV files after tests."""
        for file_name in TSV_MAPPING.values():
            os.remove(os.path.join(TSV_DIR, file_name))

if __name__ == "__main__":
    unittest.main()

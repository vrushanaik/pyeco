import unittest
import importlib.metadata
import os
import pandas as pd
import fastparquet


class TestFastparquet(unittest.TestCase):

    TEST_FILE = "test_fp_subtest.parquet"

    def tearDown(self):
        if os.path.exists(self.TEST_FILE):
            os.remove(self.TEST_FILE)

    def test_import(self):
        """Check fastparquet can be imported"""
        try:
            import fastparquet
        except ImportError:
            self.fail("fastparquet is not installed")

    def test_version(self):
        """Verify fastparquet version"""
        version = importlib.metadata.version("fastparquet")
        assert "2024.11.0" in version, f"'2024.11.0' not found in version string: {version}"

    def test_write_and_read_roundtrip(self):
        """Write a DataFrame to Parquet and read it back identically"""
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        fastparquet.write(self.TEST_FILE, df)
        df_read = fastparquet.ParquetFile(self.TEST_FILE).to_pandas()
        pd.testing.assert_frame_equal(df.reset_index(drop=True), df_read.reset_index(drop=True))

    def test_column_selection(self):
        """Read only a subset of columns"""
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "score": [10.0, 20.0]})
        fastparquet.write(self.TEST_FILE, df)
        pf = fastparquet.ParquetFile(self.TEST_FILE)
        df_partial = pf.to_pandas(columns=["id", "score"])
        self.assertListEqual(list(df_partial.columns), ["id", "score"])
        self.assertEqual(len(df_partial), 2)

    def test_append(self):
        """Appending to an existing Parquet file increases row count"""
        df1 = pd.DataFrame({"v": [1, 2]})
        df2 = pd.DataFrame({"v": [3, 4]})
        fastparquet.write(self.TEST_FILE, df1)
        fastparquet.write(self.TEST_FILE, df2, append=True)
        df_all = fastparquet.ParquetFile(self.TEST_FILE).to_pandas()
        self.assertEqual(len(df_all), 4)


if __name__ == "__main__":
    unittest.main()

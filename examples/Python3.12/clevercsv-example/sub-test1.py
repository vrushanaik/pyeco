import unittest
import importlib.metadata
import os
import clevercsv


class TestCleverCSV(unittest.TestCase):

    COMMA_FILE = "test_csv_comma.csv"
    SEMI_FILE  = "test_csv_semi.csv"

    def tearDown(self):
        for f in [self.COMMA_FILE, self.SEMI_FILE]:
            if os.path.exists(f):
                os.remove(f)

    def test_import(self):
        """Check clevercsv can be imported"""
        try:
            import clevercsv
        except ImportError:
            self.fail("clevercsv is not installed")

    def test_version(self):
        """Verify clevercsv version"""
        version = importlib.metadata.version("clevercsv")
        assert "0.8.5" in version, f"'0.8.5' not found in version string: {version}"

    def test_detect_comma_delimiter(self):
        """Sniffer should detect comma delimiter"""
        with open(self.COMMA_FILE, "w") as f:
            f.write("col1,col2,col3\n1,2,3\n4,5,6\n")
        dialect = clevercsv.Sniffer().sniff(open(self.COMMA_FILE).read(), verbose=False)
        self.assertEqual(dialect.delimiter, ",")

    def test_detect_semicolon_delimiter(self):
        """Sniffer should detect semicolon delimiter"""
        with open(self.SEMI_FILE, "w") as f:
            f.write("a;b;c\n1;2;3\n4;5;6\n")
        dialect = clevercsv.Sniffer().sniff(open(self.SEMI_FILE).read(), verbose=False)
        self.assertEqual(dialect.delimiter, ";")

    def test_read_table_row_count(self):
        """read_table should return correct number of rows (including header)"""
        with open(self.COMMA_FILE, "w") as f:
            f.write("x,y\n10,20\n30,40\n50,60\n")
        rows = clevercsv.read_table(self.COMMA_FILE)
        # 1 header row + 3 data rows
        self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()

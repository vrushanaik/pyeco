import unittest
import importlib.metadata
import lz4.frame
import lz4.block


class TestLZ4(unittest.TestCase):

    def test_import(self):
        """Check lz4 can be imported"""
        try:
            import lz4.frame
            import lz4.block
        except ImportError:
            self.fail("lz4 is not installed")

    def test_version(self):
        """Verify lz4 version"""
        version = importlib.metadata.version("lz4")
        assert "4.4.5" in version, f"'4.4.5' not found in version string: {version}"

    def test_frame_roundtrip(self):
        """lz4.frame compress/decompress round-trip"""
        data = b"hello lz4 " * 500
        compressed   = lz4.frame.compress(data)
        decompressed = lz4.frame.decompress(compressed)
        self.assertEqual(decompressed, data)

    def test_frame_compression_reduces_size(self):
        """Compressing repetitive data should shrink its size"""
        data = b"aaaa" * 1000
        compressed = lz4.frame.compress(data)
        self.assertLess(len(compressed), len(data))

    def test_block_roundtrip(self):
        """lz4.block compress/decompress round-trip"""
        data = b"block compression test " * 200
        compressed   = lz4.block.compress(data)
        decompressed = lz4.block.decompress(compressed, uncompressed_size=len(data))
        self.assertEqual(decompressed, data)

    def test_compression_levels(self):
        """Multiple compression levels should all produce valid output"""
        data = b"level test " * 300
        for level in [0, 9, 16]:
            c = lz4.frame.compress(data, compression_level=level)
            d = lz4.frame.decompress(c)
            self.assertEqual(d, data)


if __name__ == "__main__":
    unittest.main()

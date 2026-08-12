## Purpose: Demonstrates fast LZ4 compression and decompression using the lz4 library.

### Packages used:
lz4

### Functionality:

- Compresses and decompresses a text payload using lz4.frame (framing format).
- Benchmarks multiple compression levels on random data.
- Compresses and decompresses data using lz4.block (low-level block API).
- Demonstrates streaming compress/decompress via the frame API.

### How to run the example:
```
chmod +x install_test_example.sh
./install_test_example.sh
```

### License:
It's covered under Apache 2.0 licenses

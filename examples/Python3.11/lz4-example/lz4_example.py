import lz4.frame
import lz4.block
import os

print("LZ4 Example — Fast Compression and Decompression\n")

# --- 1. Basic compress / decompress with lz4.frame ---
print("1. Compress and decompress a text payload (lz4.frame)")

original = b"Hello from IBM Power! " * 100
compressed = lz4.frame.compress(original)
decompressed = lz4.frame.decompress(compressed)

ratio = len(compressed) / len(original)
print(f"   Original size    : {len(original):>6} bytes")
print(f"   Compressed size  : {len(compressed):>6} bytes")
print(f"   Compression ratio: {ratio:.3f}")
print(f"   Round-trip OK    : {decompressed == original}")

# --- 2. Different compression levels ---
print("\n2. Compression levels (lz4.frame)")
payload = os.urandom(8192)        # random bytes — hard to compress
for level in [0, 9, 16]:
    c = lz4.frame.compress(payload, compression_level=level)
    print(f"   Level {level:>2}: compressed = {len(c)} bytes")

# --- 3. lz4.block (no framing, lower-level) ---
print("\n3. Block-level compression (lz4.block)")

text = b"The quick brown fox jumps over the lazy dog. " * 50
c_block = lz4.block.compress(text)
d_block = lz4.block.decompress(c_block, uncompressed_size=len(text))

print(f"   Original   : {len(text)} bytes")
print(f"   Compressed : {len(c_block)} bytes")
print(f"   Round-trip OK: {d_block == text}")

# --- 4. Streaming with frame file-like API ---
print("\n4. Streaming compress/decompress (lz4.frame file object)")
data_chunks = [b"chunk_%04d " % i for i in range(200)]
original_stream = b"".join(data_chunks)

# Compress using context manager
buf = lz4.frame.compress(original_stream)

# Decompress
recovered = lz4.frame.decompress(buf)
print(f"   Streamed bytes  : {len(original_stream)}")
print(f"   Compressed bytes: {len(buf)}")
print(f"   Round-trip OK   : {recovered == original_stream}")

print("\nLZ4 example completed successfully!")

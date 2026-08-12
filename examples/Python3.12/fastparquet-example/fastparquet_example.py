import os
import pandas as pd
import fastparquet

print("fastparquet Example — Parquet Read/Write\n")

OUTPUT_FILE = "sample_data.parquet"

# --- 1. Create a sample DataFrame ---
print("1. Creating sample DataFrame")
df = pd.DataFrame({
    "name":       ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "department": ["Engineering", "HR", "Engineering", "Finance", "HR"],
    "salary":     [95000, 72000, 110000, 88000, 69000],
    "active":     [True, True, False, True, True],
})
print(df.to_string(index=False))

# --- 2. Write to Parquet ---
print(f"\n2. Writing DataFrame to '{OUTPUT_FILE}'")
fastparquet.write(OUTPUT_FILE, df)
size = os.path.getsize(OUTPUT_FILE)
print(f"   File size: {size} bytes")

# --- 3. Read back from Parquet ---
print(f"\n3. Reading '{OUTPUT_FILE}' back")
pf = fastparquet.ParquetFile(OUTPUT_FILE)
df_read = pf.to_pandas()
print(df_read.to_string(index=False))

# --- 4. Inspect file metadata ---
print("\n4. Parquet file metadata")
print(f"   Columns   : {list(pf.columns)}")
print(f"   Row groups: {pf.info()['row_groups']}")
print(f"   Schema    : {pf.schema}")

# --- 5. Selective column read ---
print("\n5. Reading only 'name' and 'salary' columns")
df_partial = pf.to_pandas(columns=["name", "salary"])
print(df_partial.to_string(index=False))

# --- 6. Append a new row group ---
print(f"\n6. Appending one extra row to '{OUTPUT_FILE}'")
df_extra = pd.DataFrame({
    "name":       ["Frank"],
    "department": ["Engineering"],
    "salary":     [105000],
    "active":     [True],
})
fastparquet.write(OUTPUT_FILE, df_extra, append=True)
df_full = fastparquet.ParquetFile(OUTPUT_FILE).to_pandas()
print(f"   Total rows after append: {len(df_full)}")

# Cleanup
os.remove(OUTPUT_FILE)

print("\nfastparquet example completed successfully!")

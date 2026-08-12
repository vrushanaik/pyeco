## Purpose: Demonstrates writing and reading Parquet files using the fastparquet library.

### Packages used:
fastparquet, pandas, numpy

### Functionality:

- Creates a sample pandas DataFrame with mixed column types.
- Writes the DataFrame to a Parquet file using fastparquet.
- Reads the Parquet file back and displays its contents.
- Inspects Parquet file metadata (columns, schema, row groups).
- Demonstrates selective column reads.
- Appends an additional row group to an existing Parquet file.

### How to run the example:
```
chmod +x install_test_example.sh
./install_test_example.sh
```

### License:
It's covered under Apache 2.0 licenses

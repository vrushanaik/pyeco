import os
import clevercsv

print("CleverCSV Example — Automatic CSV Dialect Detection\n")

# --- 1. Comma-delimited CSV ---
print("1. Detecting dialect of a standard comma-delimited CSV")
CSV_COMMA = "sample_comma.csv"
with open(CSV_COMMA, "w") as f:
    f.write("name,age,city\n")
    f.write("Alice,30,New York\n")
    f.write("Bob,25,London\n")
    f.write("Charlie,35,Paris\n")

dialect = clevercsv.Sniffer().sniff(open(CSV_COMMA).read(), verbose=False)
print(f"   Detected delimiter : {repr(dialect.delimiter)}")
print(f"   Quote char         : {repr(dialect.quotechar)}")

rows = clevercsv.read_table(CSV_COMMA)
print(f"   Rows read: {rows}")

# --- 2. Semicolon-delimited CSV ---
print("\n2. Detecting dialect of a semicolon-delimited CSV")
CSV_SEMI = "sample_semi.csv"
with open(CSV_SEMI, "w") as f:
    f.write("product;price;in_stock\n")
    f.write("Apple;1.20;yes\n")
    f.write("Banana;0.50;yes\n")
    f.write("Cherry;2.99;no\n")

dialect = clevercsv.Sniffer().sniff(open(CSV_SEMI).read(), verbose=False)
print(f"   Detected delimiter : {repr(dialect.delimiter)}")

rows = clevercsv.read_table(CSV_SEMI)
print(f"   Rows read: {rows}")

# --- 3. Tab-delimited CSV with quoted fields ---
print("\n3. Detecting dialect of a tab-delimited file with quoted fields")
CSV_TAB = "sample_tab.csv"
with open(CSV_TAB, "w") as f:
    f.write('id\ttitle\tdescription\n')
    f.write('1\t"Python"\t"A versatile language"\n')
    f.write('2\t"Z3"\t"An SMT solver"\n')

dialect = clevercsv.Sniffer().sniff(open(CSV_TAB).read(), verbose=False)
print(f"   Detected delimiter : {repr(dialect.delimiter)}")
print(f"   Quote char         : {repr(dialect.quotechar)}")

# --- 4. Read via wrappers module (pandas-like) ---
print("\n4. Reading comma CSV using clevercsv.wrappers")
table = clevercsv.read_table(CSV_COMMA)
header, *data_rows = table
print(f"   Header : {header}")
for row in data_rows:
    print(f"   Row    : {row}")

# Cleanup
for f in [CSV_COMMA, CSV_SEMI, CSV_TAB]:
    os.remove(f)

print("\nCleverCSV example completed successfully!")

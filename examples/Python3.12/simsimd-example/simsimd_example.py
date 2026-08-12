import simsimd
import numpy as np

print("SimSIMD Example — Hardware-accelerated Vector Similarity\n")

# --- 1. Cosine similarity ---
print("1. Cosine similarity between two float32 vectors")

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 5.0, 6.0], dtype=np.float32)

cos_sim = simsimd.cosine(a, b)
print(f"   a = {a}")
print(f"   b = {b}")
print(f"   Cosine distance (1 - similarity): {cos_sim:.6f}")

# --- 2. Inner (dot) product ---
print("\n2. Inner product")
inner = simsimd.inner(a, b)
print(f"   Inner product: {inner:.6f}")

# --- 3. Squared Euclidean (L2) distance ---
print("\n3. Squared Euclidean distance")
l2sq = simsimd.sqeuclidean(a, b)
print(f"   Squared L2 distance: {l2sq:.6f}")

# --- 4. Compare with NumPy reference ---
print("\n4. Verifying against NumPy reference values")
a_n = a / np.linalg.norm(a)
b_n = b / np.linalg.norm(b)
np_cosine_dist = 1.0 - float(np.dot(a_n, b_n))
np_inner       = float(np.dot(a, b))
np_l2sq        = float(np.sum((a - b) ** 2))

print(f"   NumPy cosine distance : {np_cosine_dist:.6f}  | SimSIMD: {cos_sim:.6f}")
print(f"   NumPy inner product   : {np_inner:.6f}  | SimSIMD: {inner:.6f}")
print(f"   NumPy squared L2      : {np_l2sq:.6f}  | SimSIMD: {l2sq:.6f}")

# --- 5. Batch pairwise distances ---
print("\n5. Batch cosine distances (3 query vectors vs 1 target)")
queries = np.random.rand(3, 3).astype(np.float32)
target  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
for i, q in enumerate(queries):
    dist = simsimd.cosine(q, target)
    print(f"   query[{i}] {q.round(4)} → cosine dist = {dist:.6f}")

print("\nSimSIMD example completed successfully!")

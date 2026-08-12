from z3 import Int, Solver, sat

print("Z3 Solver Example — Constraint Satisfaction\n")

# --- 1. Simple arithmetic constraint ---
print("1. Solve: x + y = 10, x - y = 4")

x = Int("x")
y = Int("y")

solver = Solver()
solver.add(x + y == 10)
solver.add(x - y == 4)

result = solver.check()
print(f"   Status : {result}")
if result == sat:
    model = solver.model()
    print(f"   x = {model[x]}, y = {model[y]}")

# --- 2. Integer bounds puzzle ---
print("\n2. Find all integers x in [1..9] where x*x < 50")

solutions = []
for val in range(1, 10):
    s = Solver()
    v = Int("v")
    s.add(v == val, v * v < 50)
    if s.check() == sat:
        solutions.append(val)
print(f"   Solutions: {solutions}")

# --- 3. Simple scheduling puzzle ---
print("\n3. Schedule 3 tasks (A, B, C) on slots 1-3, all different")

a = Int("A")
b = Int("B")
c = Int("C")

s = Solver()
for var in [a, b, c]:
    s.add(var >= 1, var <= 3)
s.add(a != b, b != c, a != c)

if s.check() == sat:
    m = s.model()
    print(f"   A=slot{m[a]}, B=slot{m[b]}, C=slot{m[c]}")

print("\nZ3 Solver example completed successfully!")

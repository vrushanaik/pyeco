import unittest
import importlib.metadata
from z3 import Int, Bool, Solver, sat, unsat


class TestZ3Solver(unittest.TestCase):

    def test_import(self):
        """Check z3-solver can be imported"""
        try:
            import z3
        except ImportError:
            self.fail("z3-solver is not installed")

    def test_version(self):
        """Verify z3-solver version"""
        version = importlib.metadata.version("z3-solver")
        assert "5.0.0.0" in version, f"'5.0.0.0' not found in version string: {version}"

    def test_sat_solution(self):
        """Solve a simple SAT problem and check model"""
        x = Int("x")
        y = Int("y")
        s = Solver()
        s.add(x + y == 10, x - y == 2)
        self.assertEqual(s.check(), sat)
        m = s.model()
        self.assertEqual(m[x].as_long() + m[y].as_long(), 10)
        self.assertEqual(m[x].as_long() - m[y].as_long(), 2)

    def test_unsat_detection(self):
        """Detect an unsatisfiable system"""
        x = Int("x")
        s = Solver()
        s.add(x > 10, x < 5)
        self.assertEqual(s.check(), unsat)

    def test_boolean_constraint(self):
        """Solve a boolean constraint"""
        p = Bool("p")
        q = Bool("q")
        s = Solver()
        s.add(p, q)
        self.assertEqual(s.check(), sat)
        m = s.model()
        self.assertTrue(bool(m[p]))
        self.assertTrue(bool(m[q]))


if __name__ == "__main__":
    unittest.main()

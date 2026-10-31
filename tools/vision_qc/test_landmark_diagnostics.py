import unittest

from .landmark_diagnostics import intersection


class CentreGeometryTest(unittest.TestCase):
    def test_centre_uses_intersection_not_midpoint_of_projected_circle(self):
        p=intersection([3,0],[5,8],[0,3],[10,3])
        self.assertAlmostEqual(p[0],3.75);self.assertAlmostEqual(p[1],3)
        self.assertNotEqual(p,[5,3])

    def test_parallel_lines_do_not_fabricate_a_centre(self):
        self.assertIsNone(intersection([0,0],[1,1],[0,1],[1,2]))

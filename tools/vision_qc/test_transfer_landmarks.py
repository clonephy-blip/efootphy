import unittest

import numpy as np

from tools.vision_qc.transfer_landmarks import geometry_gate


class GeometryGateTests(unittest.TestCase):
    def test_distributed_consistent_matches_pass(self):
        source=np.float32([[x,y] for x in [100,200,300,400] for y in [100,200,300,400]])
        matrix=np.array([[1,0,10],[0,1,15],[0,0,1]],np.float64)
        target=source+[10,15]
        self.assertTrue(geometry_gate(source,target,np.ones(16),matrix))

    def test_local_player_cluster_is_rejected(self):
        source=np.float32([[x,y] for x in [100,102,104,106] for y in [100,102,104,106]])
        self.assertFalse(geometry_gate(source,source,np.ones(16),np.eye(3)))

    def test_low_consensus_or_bad_projection_is_rejected(self):
        source=np.float32([[x,y] for x in [100,200,300,400] for y in [100,200,300,400]])
        self.assertFalse(geometry_gate(source,source,np.r_[np.ones(10),np.zeros(6)],np.eye(3)))
        self.assertFalse(geometry_gate(source,source+30,np.ones(16),np.eye(3)))


if __name__=='__main__':unittest.main()

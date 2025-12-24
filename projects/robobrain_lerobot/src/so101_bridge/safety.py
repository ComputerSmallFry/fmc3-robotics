# src/so101_bridge/safety.py
from typing import List, Optional, Tuple
import numpy as np

class JointSafety:
    def __init__(self, dof: int, joint_limits: Optional[List[Tuple[float, float]]] = None):
        self.dof = dof
        self.joint_limits = joint_limits

    def sanitize_target(self, q: List[float], max_delta: float, q_current: List[float]) -> List[float]:
        if len(q) != self.dof:
            raise ValueError(f"q length must be {self.dof}")
        qc = np.array(q_current, dtype=float)
        qt = np.array(q, dtype=float)

        # clamp delta
        delta = np.clip(qt - qc, -max_delta, max_delta)
        qt = qc + delta

        # clamp joint limits if available
        if self.joint_limits is not None:
            for i, (mn, mx) in enumerate(self.joint_limits):
                qt[i] = float(np.clip(qt[i], mn, mx))

        return qt.tolist()

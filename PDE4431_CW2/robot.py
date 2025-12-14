# robot.py

import numpy as np


def rotx(theta: float) -> np.ndarray:
    """Rotation about X axis (4x4 homogeneous)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c, -s,  0.0],
        [0.0,  s,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotz(theta: float) -> np.ndarray:
    """Rotation about Z axis (4x4 homogeneous)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def transl(x: float, y: float, z: float) -> np.ndarray:
    """Homogeneous translation (4x4)."""
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


class RTSS5Robot:
    """
    RTSS-5: Rotating Telescopic Shelf-Stacker, 5-DOF R–P–R–R–R arm.

    Joints:
        q1: base yaw         (R, about global Z)
        q2: vertical lift    (P, along global Z)
        q3: shoulder         (R, about local Z in horizontal plane)
        q4: elbow            (R, about same axis as shoulder)
        q5: wrist pitch      (R, about local X at the tool, tilt only)

    Geometry (metres):
        - Base frame at (0, 0, 0)
        - Base platform offset above floor: h0
        - Upper-arm length:        L1
        - Forearm length:          L2
        - Tool length:             L_tool
        - For analytic IK we combine L2 and L_tool into an effective L2_eff
          for planar reach in XY.
    """

    def __init__(self):
        # Geometric constants
        self.h0 = 0.20       # base offset above floor
        self.L1 = 0.35       # upper arm length
        self.L2 = 0.30       # forearm length
        self.L_tool = 0.10   # tool length

        # Effective second link length for planar IK (forearm + tool)
        self.L2_eff = self.L2 + self.L_tool

        # Joint limits
        self.q1_min = np.deg2rad(-150.0)
        self.q1_max = np.deg2rad(+150.0)

        self.q2_min = 0.00      # vertical travel (m)
        self.q2_max = 0.80

        self.q3_min = np.deg2rad(-120.0)
        self.q3_max = np.deg2rad(+120.0)

        self.q4_min = np.deg2rad(-135.0)
        self.q4_max = np.deg2rad(+135.0)

        self.q5_min = np.deg2rad(-180.0)
        self.q5_max = np.deg2rad(+180.0)

    # ---------------------------------------------------------------
    # Joint utilities
    # ---------------------------------------------------------------
    def clamp(self, q: np.ndarray) -> np.ndarray:
        """Clamp a 5-vector of joint values to limits."""
        q = np.asarray(q, dtype=float).reshape(5)
        q[0] = np.clip(q[0], self.q1_min, self.q1_max)
        q[1] = np.clip(q[1], self.q2_min, self.q2_max)
        q[2] = np.clip(q[2], self.q3_min, self.q3_max)
        q[3] = np.clip(q[3], self.q4_min, self.q4_max)
        q[4] = np.clip(q[4], self.q5_min, self.q5_max)
        return q

    def within_limits(self, q: np.ndarray) -> bool:
        """Check if all joints are within limits."""
        q = np.asarray(q, dtype=float).reshape(5)
        return (
            self.q1_min <= q[0] <= self.q1_max and
            self.q2_min <= q[1] <= self.q2_max and
            self.q3_min <= q[2] <= self.q3_max and
            self.q4_min <= q[3] <= self.q4_max and
            self.q5_min <= q[4] <= self.q5_max
        )

    # ---------------------------------------------------------------
    # Forward kinematics
    # ---------------------------------------------------------------
    def forward_kinematics(self, q: np.ndarray):
        """
        Compute joint positions and EE transform.

        Args:
            q: array-like, shape (5,) => [q1, q2, q3, q4, q5]

        Returns:
            joint_positions: [p0, p1, p2, p3, p4, p5]
                p0 = base origin
                p1 = after base yaw (still at origin)
                p2 = after vertical prismatic
                p3 = shoulder joint end (after L1)
                p4 = elbow joint end (after L2)
                p5 = end-effector origin (after tool offset)
            T_0_5: 4x4 homogeneous transform of end-effector.
        """
        q = self.clamp(q)
        q1, q2, q3, q4, q5 = q

        # Base frame
        T = np.eye(4)
        joint_positions = [T[:3, 3].copy()]  # p0

        # 1) Base yaw about Z
        T = T @ rotz(q1)
        joint_positions.append(T[:3, 3].copy())  # p1

        # 2) Vertical lift: h0 + q2 along Z
        T = T @ transl(0.0, 0.0, self.h0 + q2)
        joint_positions.append(T[:3, 3].copy())  # p2

        # 3) Shoulder: planar rotation, then link L1 along local X
        T = T @ rotz(q3) @ transl(self.L1, 0.0, 0.0)
        joint_positions.append(T[:3, 3].copy())  # p3

        # 4) Elbow: additional planar rotation, then forearm L2
        T = T @ rotz(q4) @ transl(self.L2, 0.0, 0.0)
        joint_positions.append(T[:3, 3].copy())  # p4

        # 5) Wrist: rotate about local X (tilt), then translate along local X by tool length
        T = T @ rotx(q5) @ transl(self.L_tool, 0.0, 0.0)
        joint_positions.append(T[:3, 3].copy())  # p5

        return joint_positions, T

    def end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """Return just the end-effector position (3,) from FK."""
        joints, _ = self.forward_kinematics(q)
        return joints[-1]

    # ---------------------------------------------------------------
    # Analytic IK for position (q1, q2, q3, q4) + simple q5
    # ---------------------------------------------------------------
    def ik_position(self,
                    target_pos: np.ndarray,
                    desired_yaw: float = None,
                    elbow_up: bool = True,
                    debug: bool = False):
        """
        Analytic IK for EE position (x, y, z). Orientation is simplified.

        Strategy:
            - q1 from atan2(y, x)  (base yaw)
            - q2 sets the vertical height directly: z = h0 + q2
            - (q3, q4) use 2R planar IK in the horizontal plane to reach
              r = sqrt(x^2 + y^2), with link lengths L1 and L2_eff
            - q5 is set from desired_yaw (orientation) if provided, else 0.

        Returns:
            q (np.ndarray, shape (5,)), success_flag (bool)
        """
        px, py, pz = np.asarray(target_pos, dtype=float).reshape(3)

        # Base yaw & horizontal distance
        q1 = np.arctan2(py, px)
        r = np.hypot(px, py)

        # Vertical: prismatic lift
        q2 = pz - self.h0
        if q2 < self.q2_min - 1e-6 or q2 > self.q2_max + 1e-6:
            if debug:
                print("[IK] z out of range:",
                      f"pz={pz:.3f}, q2={q2:.3f},",
                      f"limits=[{self.q2_min:.3f},{self.q2_max:.3f}]")
            q_dummy = np.array([q1,
                                np.clip(q2, self.q2_min, self.q2_max),
                                0.0, 0.0, 0.0])
            return q_dummy, False

        L1 = self.L1
        L2 = self.L2_eff  # effective second link (forearm + tool)

        # Law of cosines for elbow angle in planar 2R
        cos_raw = (r**2 - L1**2 - L2**2) / (2.0 * L1 * L2)
        cos_q4 = np.clip(cos_raw, -1.0, 1.0)

        # Check truly unreachable (beyond some tolerance)
        if abs(cos_raw) > 1.0 + 1e-6:
            if debug:
                print("[IK] radial out of reach: r=%.3f, cos_raw=%.6f" % (r, cos_raw))
            q_dummy = np.array([q1, q2, 0.0, 0.0, 0.0])
            return q_dummy, False

        base_elbow = np.arccos(cos_q4)
        q4 = base_elbow if elbow_up else -base_elbow

        # Shoulder angle
        k1 = L1 + L2 * np.cos(q4)
        k2 = L2 * np.sin(q4)
        q3 = -np.arctan2(k2, k1)

        # Wrist tilt: does not affect position in this model
        if desired_yaw is None:
            q5 = 0.0
        else:
            q5 = desired_yaw - (q1 + q3 + q4)

        q = np.array([q1, q2, q3, q4, q5], dtype=float)

        # Joint limits
        if not self.within_limits(q):
            if debug:
                print("[IK] result outside joint limits:", q)
            return q, False

        # Verify final position error
        ee_pos = self.end_effector_position(q)
        err = np.linalg.norm(ee_pos - target_pos)
        success = (err < 5e-3)   # 5 mm tolerance

        if debug:
            print(f"[IK] target={target_pos}, ee={ee_pos}, err={err:.3e}, success={success}")

        return q, success

    # ---------------------------------------------------------------
    # Workspace sampling
    # ---------------------------------------------------------------
    def sample_workspace(self, num_samples: int = 800):
        """Randomly sample joint space and return EE points as (N,3)."""
        q1 = np.random.uniform(self.q1_min, self.q1_max, size=num_samples)
        q2 = np.random.uniform(self.q2_min, self.q2_max, size=num_samples)
        q3 = np.random.uniform(self.q3_min, self.q3_max, size=num_samples)
        q4 = np.random.uniform(self.q4_min, self.q4_max, size=num_samples)
        q5 = np.random.uniform(self.q5_min, self.q5_max, size=num_samples)

        pts = []
        for i in range(num_samples):
            q = np.array([q1[i], q2[i], q3[i], q4[i], q5[i]])
            pts.append(self.end_effector_position(q))
        return np.vstack(pts)

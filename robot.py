import numpy as np

def rotx(theta: float) -> np.ndarray:
    """Rotation about X axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c, -s,  0.0],
        [0.0,  s,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

# def rotx(theta: float) -> np.ndarray:
#     """Rotation about Y axis."""
#     c, s = np.cos(theta), np.sin(theta)
#     return np.array([
#         [1.0, 0.0, 0.0, 0.0],
#         [0.0,  c, -s,  0.0],
#         [0.0,  s,  c,  0.0],
#         [0.0, 0.0, 0.0, 1.0],
#     ])

def rotz(theta: float) -> np.ndarray:
    """Rotation about Z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def transl(x: float, y: float, z: float) -> np.ndarray:
    """Homogeneous translation."""
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
        - For IK we combine L2 and L_tool into an effective L2_eff.
    """

    def __init__(self):
        # Geometric constants
        self.h0 = 0.20       # base offset above floor
        self.L1 = 0.35       # upper arm length
        self.L2 = 0.30       # forearm length
        self.L_tool = 0.10   # tool length

        self.L2_eff = self.L2 + self.L_tool  # used in planar IK

        # Joint limits
        self.q1_min = np.deg2rad(-150.0)
        self.q1_max = np.deg2rad(+150.0)

        self.q2_min = 0.00      # vertical travel
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

        Returns:
            joint_positions: [p0, p1, p2, p3, p4, p5]
                p0 = base origin
                p1 = after base yaw (still at origin)
                p2 = after vertical prismatic
                p3 = shoulder joint
                p4 = elbow joint
                p5 = end-effector origin
            T_0_5: 4x4 homogeneous transform of end-effector.
        """
        q = self.clamp(q)
        q1, q2, q3, q4, q5 = q

        # Base
        T = np.eye(4)
        joint_positions = [T[:3, 3].copy()]  # p0

        # Base yaw
        T = T @ rotz(q1)
        joint_positions.append(T[:3, 3].copy())  # p1

        # Vertical lift
        T = T @ transl(0.0, 0.0, self.h0 + q2)
        joint_positions.append(T[:3, 3].copy())  # p2

        # Shoulder: rotate in horizontal plane, then link L1 along local X
        T = T @ rotz(q3) @ transl(self.L1, 0.0, 0.0)
        joint_positions.append(T[:3, 3].copy())  # p3 (shoulder link end)

        # Elbow: additional rotation and forearm length L2
        T = T @ rotz(q4) @ transl(self.L2, 0.0, 0.0)
        joint_positions.append(T[:3, 3].copy())  # p4 (elbow link end)

        # Wrist + tool:
        #   - rotate about local X by q5 (tilt)
        #   - then translate along local X by L_tool
        T = T @ rotx(q5) @ transl(self.L_tool, 0.0, 0.0)
        joint_positions.append(T[:3, 3].copy())  # p5 (Wrist link)


        return joint_positions, T

    def end_effector_position(self, q: np.ndarray) -> np.ndarray:
        joints, _ = self.forward_kinematics(q)
        return joints[-1]

    # ---------------------------------------------------------------
    # Analytic IK for position (q1, q2, q3, q4) + simple q5
    # ---------------------------------------------------------------
    def ik_position(self, target_pos: np.ndarray, desired_yaw: float = None,
                    elbow_up: bool = True):
        """
        Analytic IK for EE position (x, y, z). Orientation is simplified.

        Strategy:
            - q1 from atan2(y, x)
            - q2 sets the vertical height directly: z = h0 + q2
            - (q3, q4) use 2R planar IK in the horizontal plane to reach
              the radial distance r = sqrt(x^2 + y^2), with link lengths
              L1 and L2_eff = L2 + L_tool
            - q5 set from desired_yaw if given, else 0.

        Returns:
            q, success_flag
        """
        px, py, pz = np.asarray(target_pos, dtype=float).reshape(3)

        # Base yaw (direction to target in XY)
        q1 = np.arctan2(py, px)

        # Radial distance from base axis
        r = np.hypot(px, py)

        # Vertical: use prismatic to match the requested z
        q2 = pz - self.h0

        # Clamp q2 and check if it's feasible
        if q2 < self.q2_min or q2 > self.q2_max:
            # still build q for debugging, but mark as failure
            q_dummy = np.array([q1, np.clip(q2, self.q2_min, self.q2_max), 0.0, 0.0, 0.0])
            return q_dummy, False

        # 2R planar IK in the horizontal plane for (r, 0)
        L1 = self.L1
        L2 = self.L2_eff

        # Law of cosines for elbow angle
        cos_q4 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)

        if cos_q4 < -1.0 or cos_q4 > 1.0:
            # Target out of reach in XY
            q_dummy = np.array([q1, q2, 0.0, 0.0, 0.0])
            return q_dummy, False

        base_elbow = np.arccos(np.clip(cos_q4, -1.0, 1.0))
        q4 = base_elbow if elbow_up else -base_elbow

        # Shoulder angle
        k1 = L1 + L2 * np.cos(q4)
        k2 = L2 * np.sin(q4)
        # Target in local planar frame is at (x=r, y=0)
        # atan2(0, r) = 0
        q3 = 0.0 - np.arctan2(k2, k1)

        # Wrist tilt
        if desired_yaw is None: #interpret desired_yaw as a desired pitch/tilt angle about X
            q5 = 0.0
        else:
            # Simple model: global yaw approx q1 + q3 + q4 + q5
            q5 = desired_yaw - (q1 + q3 + q4)

        q = np.array([q1, q2, q3, q4, q5], dtype=float)

        # Check limits and verify position error
        if not self.within_limits(q):
            return q, False

        ee_pos = self.end_effector_position(q)
        err = np.linalg.norm(ee_pos - target_pos)

        success = (err < 1e-3)
        return q, success

    # ---------------------------------------------------------------
    # Workspace sampling
    # ---------------------------------------------------------------
    def sample_workspace(self, num_samples: int = 800):
        """Randomly sample joint space and return EE points."""
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

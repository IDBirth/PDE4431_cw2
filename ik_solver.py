import numpy as np


def rotation_matrix_log(R: np.ndarray) -> np.ndarray:
    """
    Compute the so(3) logarithm (axis-angle vector) of a 3x3 rotation matrix.

    Returns a 3D vector w such that exp([w]x) = R.
    If R is close to identity, uses a first-order approximation.
    """
    # Ensure proper shape
    R = np.asarray(R, dtype=float).reshape(3, 3)

    # Numerical safety: clamp trace
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < 1e-6:
        # Very small angle: use approximation w ≈ 0.5 * (R - R.T)_vee
        w = 0.5 * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
        return w

    # Standard formula: w = theta / (2 sin theta) * (R - R^T)_vee
    factor = theta / (2.0 * np.sin(theta))
    w = factor * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return w


def finite_difference_jacobian(robot,
                               q: np.ndarray,
                               eps: float = 1e-4,
                               position_only: bool = True,
                               target_dim: int = 3) -> np.ndarray:
    """
    Numerically estimate Jacobian of the robot's end-effector pose
    with respect to joint variables.

    Args:
        robot: object with forward_kinematics(q) -> (joint_positions, T)
        q: current joint configuration (n,)
        eps: finite difference step
        position_only: if True, Jacobian maps to position (3D) only.
                       if False, maps to pose error (position+orientation).
        target_dim: dimension of the task space vector (3 or 6).

    Returns:
        J: (target_dim x n) approximate Jacobian.
    """
    q = np.asarray(q, dtype=float).flatten()
    n = q.size

    # Helper to get 3D position or 6D pose vector from T
    def pose_vector(T):
        p = T[:3, 3]
        if position_only:
            return p
        else:
            R = T[:3, :3]
            w = rotation_matrix_log(R)  # orientation error (axis-angle)
            return np.concatenate([p, w])

    _, T0 = robot.forward_kinematics(q)
    f0 = pose_vector(T0)

    J = np.zeros((target_dim, n), dtype=float)

    for i in range(n):
        dq = np.zeros_like(q)
        dq[i] = eps
        q_pert = q + dq

        # Optionally clamp to joint limits if available
        if hasattr(robot, "clamp"):
            q_pert = robot.clamp(q_pert)

        _, Ti = robot.forward_kinematics(q_pert)
        fi = pose_vector(Ti)

        J[:, i] = (fi - f0) / eps

    return J


class NumericalIKSolver:
    """
    Generic numerical IK solver using damped least squares (Levenberg–Marquardt style).

    Works with arbitrary DOF robot models (4, 5, 6, ...), as long as the
    robot provides:
        - forward_kinematics(q) -> (joint_positions, T)
        - clamp(q)  [optional but recommended]
        - within_limits(q)  [optional]

    Supports:
        - Position-only IK (3D)
        - Position + orientation IK (6D) if target_rot is provided.
    """

    def __init__(self,
                 robot,
                 max_iters: int = 100,
                 tol_pos: float = 1e-3,
                 tol_step: float = 1e-4,
                 damping: float = 1e-3,
                 position_only: bool = True):
        """
        Args:
            robot: robot model instance.
            max_iters: maximum IK iterations.
            tol_pos: tolerance for task-space error norm.
            tol_step: tolerance for joint update norm.
            damping: damping factor (lambda) for DLS.
            position_only: if True, solve for position only (3D task).
                           if False, solve for position + orientation (6D).
        """
        self.robot = robot
        self.max_iters = max_iters
        self.tol_pos = tol_pos
        self.tol_step = tol_step
        self.damping = damping
        self.position_only = position_only

    def solve(self,
              target_pos: np.ndarray,
              target_rot: np.ndarray = None,
              q_init: np.ndarray = None):
        """
        Solve IK for the robot.

        Args:
            target_pos: desired EE position (3,)
            target_rot: desired rotation matrix (3x3). If provided and
                        position_only=False, orientation is also matched.
            q_init: initial guess for joint configuration (n,). If None,
                    a zero vector is used.

        Returns:
            q_sol: solution joint vector (n,)
            success: True if converged, False otherwise.
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)

        if self.position_only is False:
            if target_rot is None:
                raise ValueError("target_rot must be provided when position_only=False")
            target_rot = np.asarray(target_rot, dtype=float).reshape(3, 3)

        # Determine DOF from initial guess or fallback
        if q_init is None:
            # Try to infer n from robot.clamp or robot.forward_kinematics
            # Here we default to 6-DOF; user should usually pass q_init.
            n = 6
            q = np.zeros(n, dtype=float)
        else:
            q = np.asarray(q_init, dtype=float).flatten()
            n = q.size

        # Optionally clamp initial guess
        if hasattr(self.robot, "clamp"):
            q = self.robot.clamp(q)

        def pose_error(q_vec):
            """Compute task-space error vector."""
            _, T = self.robot.forward_kinematics(q_vec)
            p = T[:3, 3]
            e_pos = target_pos - p

            if self.position_only:
                return e_pos

            # Orientation error: log(R_des^T R_cur)
            R_cur = T[:3, :3]
            R_err = target_rot @ R_cur.T
            e_ori = rotation_matrix_log(R_err)
            return np.concatenate([e_pos, e_ori])

        target_dim = 3 if self.position_only else 6

        for _ in range(self.max_iters):
            e = pose_error(q)
            err_norm = np.linalg.norm(e)

            # Check position/orientation error convergence
            if err_norm < self.tol_pos:
                break

            # Jacobian via finite differences
            J = finite_difference_jacobian(
                self.robot,
                q,
                eps=1e-4,
                position_only=self.position_only,
                target_dim=target_dim,
            )

            # Damped least squares: dq = J^T (J J^T + λ^2 I)^-1 e
            JT = J.T
            JJt = J @ JT
            lam2_I = (self.damping ** 2) * np.eye(target_dim)
            dq = JT @ np.linalg.solve(JJt + lam2_I, e)

            # Update joints
            q_new = q + dq

            # Clamp to limits if method exists
            if hasattr(self.robot, "clamp"):
                q_new = self.robot.clamp(q_new)

            step_norm = np.linalg.norm(q_new - q)
            q = q_new

            # Check step size convergence
            if step_norm < self.tol_step:
                break

        # Final error check
        e_final = pose_error(q)
        final_norm = np.linalg.norm(e_final)

        # Optionally check limits
        success_limits = True
        if hasattr(self.robot, "within_limits"):
            success_limits = bool(self.robot.within_limits(q))

        success = (final_norm < self.tol_pos) and success_limits
        return q, success

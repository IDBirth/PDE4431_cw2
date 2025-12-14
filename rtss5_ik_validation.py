# rtss5_ik_validation.py

import numpy as np

from robot import RTSS5Robot
from ik_solver import NumericalIKSolver


def validate_rtss5_single_target(target_pos: np.ndarray):
    """
    Compare analytic IK (robot.ik_position) and numerical IK (NumericalIKSolver)
    for a single target position on the RTSS-5 robot.
    """
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    robot = RTSS5Robot()

    print("--------------------------------------------------")
    print("Target position (m):", target_pos)

    # --------------------------------------------------
    # 1) Analytic IK from RTSS5Robot.ik_position
    # --------------------------------------------------
    q_analytic, ok_a = robot.ik_position(target_pos, desired_yaw=None, elbow_up=True)
    print("\nAnalytic IK:")
    print("  success:", ok_a)
    print("  q_analytic (rad):", q_analytic)

    ee_a = robot.end_effector_position(q_analytic)
    err_a = np.linalg.norm(ee_a - target_pos)
    print("  FK(analytic) (m):", ee_a)
    print(f"  ‖target - FK(analytic)‖ = {err_a:.6e} m")

    # --------------------------------------------------
    # 2) Numerical IK using the generic DLS solver
    # --------------------------------------------------
    ik_solver = NumericalIKSolver(
        robot,
        max_iters=100,
        tol_pos=1e-3,
        tol_step=1e-4,
        damping=1e-3,
        position_only=True,  # only x,y,z for this robot in CW2
    )

    # Use analytic solution as a good initial guess (or use q_home if you prefer)
    q_init = q_analytic.copy()

    q_num, ok_n = ik_solver.solve(
        target_pos=target_pos,
        target_rot=None,   # position-only mode
        q_init=q_init,
    )

    print("\nNumerical IK (DLS):")
    print("  success:", ok_n)
    print("  q_numerical (rad):", q_num)

    ee_n = robot.end_effector_position(q_num)
    err_n = np.linalg.norm(ee_n - target_pos)
    print("  FK(numerical) (m):", ee_n)
    print(f"  ‖target - FK(numerical)‖ = {err_n:.6e} m")

    # --------------------------------------------------
    # 3) Compare the two joint solutions
    # --------------------------------------------------
    dq = q_num - q_analytic
    print("\nDifference between numerical and analytic joints:")
    print("  dq (rad):", dq)
    print("  ‖dq‖:", np.linalg.norm(dq))

    print("--------------------------------------------------\n")


def sample_and_validate(num_tests: int = 5):
    """
    Run multiple random tests within a reasonable reachable region
    to validate numerical IK against analytic IK.
    """
    robot = RTSS5Robot()

    for i in range(num_tests):
        # Simple random target in a plausible reachable region:
        # radius ~ [0.2, 0.7] m, angle ~ [-60°, 60°], height ~ [0.25, 0.75] m
        radius = np.random.uniform(0.2, 0.7)
        angle = np.random.uniform(np.deg2rad(-60.0), np.deg2rad(60.0))
        z = np.random.uniform(0.25, 0.75)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        target_pos = np.array([x, y, z])

        print(f"Test {i+1}/{num_tests}")
        validate_rtss5_single_target(target_pos)


if __name__ == "__main__":
    # Example 1: explicit target (e.g. front shelf-like position)
    target_example = np.array([0.6, 0.0, 0.5])
    validate_rtss5_single_target(target_example)

    # Example 2: run a few random tests
    sample_and_validate(num_tests=3)

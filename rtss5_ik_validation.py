# rtss5_ik_validation.py

import numpy as np

from PDE4431_CW2.robot import RTSS5Robot
from PDE4431_CW2.ik_solver import NumericalIKSolver


def validate_rtss5_single_target(target_pos: np.ndarray, debug: bool = False):
    """
    Compare analytic IK (RTSS5Robot.ik_position) and numerical IK (NumericalIKSolver)
    for a single target position on the RTSS-5 robot.

    Prints:
      - target position
      - analytic IK solution + FK error
      - numerical IK solution + FK error
      - difference between the two joint solutions
    """
    target_pos = np.asarray(target_pos, dtype=float).reshape(3)
    robot = RTSS5Robot()

    print("--------------------------------------------------")
    print("Target position (m):", target_pos)

    # --------------------------------------------------
    # 1) Analytic IK from RTSS5Robot.ik_position
    # --------------------------------------------------
    q_analytic, ok_a = robot.ik_position(
        target_pos,
        desired_yaw=None,
        elbow_up=True,
        debug=debug,
    )
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
        position_only=True,  # only x, y, z for this CW
    )

    # Use analytic solution as a good initial guess if it reported success,
    # otherwise fall back to a simple "home" guess.
    if ok_a:
        q_init = q_analytic.copy()
    else:
        # Home: yaw = 0, lift = mid, planar arm straight, wrist neutral
        q_init = np.array([0.0, 0.3, 0.0, 0.0, 0.0], dtype=float)

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
    if q_analytic.shape == q_num.shape:
        dq = q_num - q_analytic
        print("\nDifference between numerical and analytic joints:")
        print("  dq (rad):", dq)
        print("  ‖dq‖:", np.linalg.norm(dq))
    else:
        print("\nCannot compare joint vectors: shapes differ.",
              q_analytic.shape, "vs", q_num.shape)

    print("--------------------------------------------------\n")


def sample_and_validate(num_tests: int = 5, debug_on_fail: bool = False):
    """
    Run multiple random tests within a reasonable reachable region
    to validate numerical IK against analytic IK.

    Sampling region (heuristic, based on RTSS-5 geometry):
      - radius r in [0.2, 0.7] m
      - angle in [-60°, 60°]
      - height z in [0.25, 0.75] m
    """
    robot = RTSS5Robot()

    for i in range(num_tests):
        # Sample a random target in a plausible reachable region
        radius = np.random.uniform(0.2, 0.7)
        angle = np.random.uniform(np.deg2rad(-60.0), np.deg2rad(60.0))
        z = np.random.uniform(0.25, 0.75)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        target_pos = np.array([x, y, z])

        print(f"Test {i+1}/{num_tests}")

        # First run without debug prints
        validate_rtss5_single_target(target_pos, debug=False)

        # Optionally re-run with debug if analytic failed badly
        if debug_on_fail:
            # Quick check: see what analytic does in a "silent" call
            q_a, ok_a = robot.ik_position(target_pos, desired_yaw=None, elbow_up=True, debug=False)
            ee_a = robot.end_effector_position(q_a)
            err_a = np.linalg.norm(ee_a - target_pos)

            if (not ok_a) or (err_a > 1e-2):
                print("[DEBUG] Re-running this test with detailed IK prints...")
                validate_rtss5_single_target(target_pos, debug=True)


if __name__ == "__main__":
    # Example 1: explicit target (e.g. front shelf-like position)
    target_example = np.array([0.6, 0.0, 0.5])
    validate_rtss5_single_target(target_example, debug=True)

    # Example 2: run a few random tests
    sample_and_validate(num_tests=3, debug_on_fail=True)

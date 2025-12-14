# RTSS-5 Shelf Stacking Manipulator (PDE4431 CW2)

Python implementation of the **RTSS-5** (Rotating Telescopic Shelf-Stacker), a
5-DOF **R–P–R–R–R** arm that stacks three objects from a tabletop onto three
front shelves. The project includes analytic IK, an optional numerical IK
solver, and a Matplotlib-based simulator with simple controls.

## Contents

- `PDE4431_CW2/robot.py` — RTSS-5 model, FK, analytic position IK, limit checks,
  workspace sampler.
- `PDE4431_CW2/simulation.py` — pick-and-place simulation, GUI controls,
  optional numerical IK, visual debugging overlays, shelf/table geometry.
- `PDE4431_CW2/ik_solver.py` — generic damped-least-squares IK helper
  (finite-difference Jacobian; position-only by default).
- `PDE4431_CW2/main.py` — entrypoint to launch the simulator.
- `rtss5_ik_validation.py` — compares analytic IK vs. numerical IK on random
  targets.
- `requirements.txt` — pinned dependencies (NumPy <2 to avoid ABI issues).

## Robot model (RTSS-5)

- Joints: `q1` base yaw (R), `q2` vertical lift (P), `q3` shoulder (R),
  `q4` elbow (R), `q5` wrist pitch (R).
- Geometry (m): base offset `h0=0.20`, upper arm `L1=0.35`, forearm `L2=0.30`,
  tool `L_tool=0.10` (used in IK as `L2_eff = L2 + L_tool`).
- Analytic IK (position-only): yaw from `atan2(y,x)`, z via `q2`, planar 2R IK
  for `(q3,q4)`, simple `q5` (zero or to track desired yaw). Limit checks and
  FK validation ensure feasibility.
- Numerical IK: toggle in `simulation.py` with `self.use_numerical_ik = True`
  to use the damped-least-squares solver (`ik_solver.NumericalIKSolver`) with
  position-only task space.

## Task layout

- Table (acts as floor): center `(0.0, -0.30, 0.0)`, size `0.4 × 0.4` m,
  height `0.22` m. Three objects are placed along X on the tabletop.
- Shelves: radius `0.70` m at `θ=30°`; heights `[0.30, 0.55, 0.80]` m. Each
  shelf is drawn as a thin platform and has a star marker target; objects carry
  a `target` pointing to their shelf.

## Simulator features

- Buttons: **Run full sequence** (pick & place all three), **Show workspace**
  (scatter sampled FK points), **Reset** (home pose and reset objects).
- IK debug overlays: goal vs. FK(q) and error norm update after each IK call.
- EE trail, object markers, shelf/table geometry drawn in 3D.
- Pick/place steps: pre-pick → pick → lift → pre-place → place → exit →
  home; objects attach/detach to EE accordingly.

## Setup & run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python PDE4431_CW2/main.py
```

> Note: NumPy is pinned to `<2` because some modules are not yet built against
> NumPy 2.x. If you see ABI errors, ensure the pinned version is installed.

## IK validation script

```bash
source .venv/bin/activate
python rtss5_ik_validation.py
```

This reports analytic vs. numerical IK solutions and their FK errors for random
targets.

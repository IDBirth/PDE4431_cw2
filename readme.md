# PDE4431 CW2 — Industrial Manipulator Kinematics

**Student:** Bilal Baslar (M01099599)  
**Robot:** RTSS-5 (Telescopic SCARA Shelf-Stacker) — 5-DOF **R–P–R–R–R** arm
that moves three objects from a tabletop “floor” to three shelves at different
heights using purely simulated kinematics.

## How this meets the Rubric

- Uses at five joints with one prismatic joint: base yaw (R), vertical
  lift (P), shoulder (R), elbow (R), wrist pitch (R).
- Denavit–Hartenberg parameters defined for each joint; PLUS analytic FK/IK validate
  reachability, with optional numerical IK (damped least squares).
- Simulation demonstrates all four required positions: three floor pickups on a
  table surface and three shelf placements at increasing heights and a fixed
  radius; workspace scatter plot confirms coverage.
- Matplotlib GUI animates the full pick-and-place sequence; objects attach to
  the end-effector, detach on shelves, and a trail shows the EE path. Buttons
  allow auto-run, workspace plotting, and reset.
- Extensibility for distinction criteria: inverse kinematics is implemented
  both analytically and numerically; joint limits, DH info overlay, and error
  reporting show validation beyond the minimum.

## Robot model and DH parameters

- Geometry (m): base offset `h0=0.20`, upper arm `L1=0.35`, forearm `L2=0.30`,
  tool `L_tool=0.10` (`L2_eff = L2 + L_tool` for planar IK).
- Current DH table (drawn in the sim overlay):
  - J1: a=0.00, α=0°,   d=h0,   θ=q1
  - J2: a=0.00, α=0°,   d=q2,   θ=0
  - J3: a=L1,  α=0°,    d=0,    θ=q3
  - J4: a=L2,  α=0°,    d=0,    θ=q4
  - J5: a=L_tool, α=0°, d=0,    θ=q5

## Task layout in simulation

- Table-as-floor: center `(0.0, -0.30, 0.0)`, size `0.4 × 0.4` m, height
  `0.22` m; three objects along X on the tabletop.
- Shelves: radius `0.70` m at `θ=30°`; heights `[0.30, 0.55, 0.80]` m; drawn
  as thin platforms with target markers.
- Home pose: `[0.0, 0.20, 0.0, 0.0, 0.0]` (yaw, lift, shoulder, elbow, wrist).

## Repository map

- `PDE4431_CW2/robot.py` — RTSS-5 model, FK, analytic position IK, limit checks,
  workspace sampler.
- `PDE4431_CW2/simulation.py` — pick-and-place loop, Matplotlib GUI, DH/IK
  overlays, workspace plotter.
- `PDE4431_CW2/ik_solver.py` — damped-least-squares IK (finite-difference
  Jacobian) for position or pose.
- `PDE4431_CW2/main.py` — simulator entrypoint.
- `rtss5_ik_validation.py` — compares analytic vs numerical IK on random targets.
- `requirements.txt` — dependency pins (NumPy <2 to avoid ABI issues).

## Run the simulator

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python PDE4431_CW2/main.py
```

- **Run full sequence:** animates pick → place for all three objects.
- **Show workspace:** scatters sampled FK points to visualise reachable set.
- **Reset:** return to home pose and reset object states.
- Toggle `use_numerical_ik` in `PDE4431_CW2/simulation.py` to compare analytic
  vs numerical IK.

## IK validation (text-only)

```bash
source .venv/bin/activate
python rtss5_ik_validation.py
```

Reports analytic and numerical IK solutions plus FK error norms for random
targets.

## Video link (per brief)

- YouTube demo (with commentary): **<add link here>**

## Submission notes

- Coursework: PDE4431 Dubai Coursework 2 — Industrial Manipulator Kinematics
  Modelling (Dec 11, 2025, 23:59 Dubai time).
- Upload code to GitHub and add `@judhi` as collaborator as required in the
  PDF instructions. Include the video link above in the GitHub README.

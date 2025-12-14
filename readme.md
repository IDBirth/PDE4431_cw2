# PDE4431 CW2 — Industrial Manipulator Kinematics

**Student:** Bilal Baslar (M01099599)  
**Robot:** RTSS-5 (Rotating Telescopic Shelf-Stacker) like a Telescopic SCARA robot — 5-DOF **R–P–R–R–R** arm
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

- **q₁** – Base yaw (revolute, about global Z)
- **q₂** – Vertical lift (prismatic, along global Z)
- **q₃** – Shoulder (revolute, about local Z)
- **q₄** – Elbow   (revolute, about local Z)
- **q₅** – Wrist pitch (revolute, about local X at the tool)

Link geometry (all in metres):

- Base offset:      h₀   = 0.20 m
- Upper arm length: L₁   = 0.35 m
- Forearm length:   L₂   = 0.30 m
- Tool length:      L_tool = 0.10 m

---

### Standard DH Table (Frames 0 → 4)

Using Craig’s standard DH convention:

Tᵢ⁽ⁱ⁺¹⁾ = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)

With the RTSS-5 geometry, the first four joints map to the following
DH parameters:

| Link i | Joint type | aᵢ [m] | αᵢ [deg] |      dᵢ [m]       |   θᵢ [deg]   |
|--------|-----------:|:------:|:--------:|:-----------------:|:------------:|
| 1      |    R (q₁)  | 0.00   |   0.0    |       0.00        |     q₁       |
| 2      |    P (q₂)  | 0.00   |   0.0    |  0.20 + q₂ (var)  |     0.0      |
| 3      |    R (q₃)  | 0.35   |   0.0    |       0.00        |     q₃       |
| 4      |    R (q₄)  | 0.30   |   0.0    |       0.00        |     q₄       |
| 5      |    R (q⁵)  | 0.30   |   0.0    |       0.00        |     q⁵       |

So the homogeneous transform up to the forearm frame (frame 4) is:

T₀⁴ = T₀¹ · T₁² · T₂³ · T₃⁴

![Robot Image](/images/Robot.png)


## Task layout in simulation

- Table-as-floor: center `(0.0, -0.30, 0.0)`, size `0.4 × 0.4` m, height `0.22` m; 
  three objects along X on the tabletop.
- Shelves: radius `0.70` m at `θ=30°`; 
  heights `[0.30, 0.55, 0.80]` m; 
  drawn as thin platforms with target markers.
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
- Uploaded code to GitHub and added `@judhi` as collaborator as required in the
  PDF instructions. Include the video link above.

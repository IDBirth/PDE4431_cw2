# RTSS-4 Shelf Stacking Manipulator (PDE4431 CW2)

This project implements a 4-DOF **R–P–P–R** cylindrical robot, the **RTSS-4
(Rotating Telescopic Shelf-Stacker)**, entirely in Python. The manipulator
is modelled using Denavit–Hartenberg style kinematics and simulates a
pick-and-place task: moving three objects from the floor onto three
shelves at different heights.

## 1. Robot Description

- 4 DOF, including **two prismatic** joints:
  - q1: base yaw (revolute about global Z)
  - q2: vertical lift (prismatic along Z)
  - q3: horizontal telescopic extension (prismatic along arm X)
  - q4: wrist yaw (revolute about local Z)

- Geometric parameters:
  - Base height above floor: `h0 = 0.20 m`
  - Fixed radial offset to telescopic stage: `r0 = 0.30 m`
  - Tool length: `L_tool = 0.10 m`

### DH-style Parameters

The robot can be described with the following DH table (standard DH,
with a mixture of d- and a-type joint variables):

| i | Type | aᵢ               | αᵢ | dᵢ        | θᵢ      |
|---|------|------------------|----|-----------|---------|
| 1 | R    | 0                | 0  | h0        | q1      |
| 2 | P    | 0                | 0  | q2        | 0       |
| 3 | P    | r0 + q3          | 0  | 0         | 0       |
| 4 | R    | L_tool           | 0  | 0         | q4      |

Forward kinematics in the code is implemented via homogeneous transforms
using these parameters.

## 2. Inverse Kinematics

Analytical IK is implemented for position:

Given target (x, y, z),

1. Base yaw:
   - q1 = atan2(y, x)

2. Radial distance:
   - r = sqrt(x² + y²)

3. Telescopic extension:
   - r = r0 + q3 + L_tool  ⇒  q3 = r − r0 − L_tool

4. Vertical lift:
   - z = h0 + q2  ⇒  q2 = z − h0

5. Wrist yaw:
   - q4 is chosen as 0, or adjusted to track a desired global yaw.

Joint limits are enforced and each IK solution is validated by FK
(position error check).

## 3. Task and Simulation

- Three floor targets:
  - Radius R = 0.70 m, small angle offsets around 30°
  - Height z = 0.05 m (near floor)
- Three shelf targets:
  - Same radius, angle 30°
  - Heights z = [0.30, 0.55, 0.80] m

The simulation performs:

- Pre-pick (above floor target)
- Pick (down to floor target)
- Pre-place (above shelf)
- Place (down to shelf)
- Return to home

The script animates the robot, the end-effector path, and the objects
moving from floor to shelves.

## 4. Running the Code

Requirements:

- Python 3.8+
- `numpy`
- `matplotlib`

Install:

```bash
pip install numpy matplotlib

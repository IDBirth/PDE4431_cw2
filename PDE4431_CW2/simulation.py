import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from robot import RTSS5Robot
from ik_solver import NumericalIKSolver



class ShelfStackingSimulation:
    """
    Simulation of the RTSS-5 robot performing pick & place:
    - Three objects on the floor (same radial distance, slightly different angles)
    - Three shelf positions at increasing heights, same radial distance.
    """

    def __init__(self):
        self.robot = RTSS5Robot()

        # -------------------------------------------------
        # IK configuration
        # -------------------------------------------------
        # Toggle this flag to switch between analytic and numerical IK
        self.use_numerical_ik = False   # set True to use NumericalIKSolver

        if self.use_numerical_ik:
            self.ik_solver = NumericalIKSolver(
                self.robot,
                max_iters=80,
                tol_pos=1e-3,
                tol_step=1e-4,
                damping=1e-3,
                position_only=True,   # we only care about (x, y, z) in this CW
            )
        else:
            self.ik_solver = None  # not needed when using analytic IK

        # -------------------------------------------------
        # Small floor table definition (used as "floor")
        # -------------------------------------------------
        self.table_center = np.array([0.0, -0.30, 0.0])  # Position on the (x, y, z) at floor
        #Size of table:
        self.table_width = 0.4   # along X
        self.table_depth = 0.4   # along Y
        self.table_height = 0.22  # along Z

        # -------------------------------------------------
        # Task geometry
        # -------------------------------------------------
        self.R_shelf = 0.70  # radius for shelves ONLY now
        self.theta_base = np.deg2rad(30.0)  # central angle for shelves

        # Table top is the new "floor" for the objects
        self.z_floor = self.table_center[2] + self.table_height  # 0.2 m
        table_x0, table_y0, _ = self.table_center
        dx = self.table_width / 2.0 * 0.6  # keep objects somewhat inside edges

        # Three balls on the table top, in a row along X
        self.floor_positions = [
            np.array([table_x0 - dx, table_y0, self.z_floor]),
            np.array([table_x0,       table_y0, self.z_floor]),
            np.array([table_x0 + dx, table_y0, self.z_floor]),
        ]

        # -------------------------------------------------
        # Shelves: 3 target locations in front of robot
        # -------------------------------------------------
        self.z_shelves = [0.30, 0.55, 0.80]   # heights of each shelf
        self.shelf_positions = []
        self.shelf_poly_verts = []
        self.shelf_polys = []

        shelf_width = 0.35      # along X
        shelf_depth = 0.20      # along Y
        shelf_thickness = 0.02  # vertical thickness

        for i, z_shelf in enumerate(self.z_shelves):
            angle = self.theta_base
            x = self.R_shelf * np.cos(angle)
            y = self.R_shelf * np.sin(angle)
            pos = np.array([x, y, z_shelf])
            self.shelf_positions.append(pos)

            # --- graphical shelf as a thin rectangular platform ---
            x0, y0, z0 = pos
            half_w = shelf_width / 2.0
            half_d = shelf_depth / 2.0

            z_bottom = z0 - shelf_thickness
            z_top = z0

            p0 = [x0 - half_w, y0 - half_d, z_bottom]
            p1 = [x0 + half_w, y0 - half_d, z_bottom]
            p2 = [x0 + half_w, y0 + half_d, z_bottom]
            p3 = [x0 - half_w, y0 + half_d, z_bottom]

            p4 = [x0 - half_w, y0 - half_d, z_top]
            p5 = [x0 + half_w, y0 - half_d, z_top]
            p6 = [x0 + half_w, y0 + half_d, z_top]
            p7 = [x0 - half_w, y0 + half_d, z_top]

            verts = [
                [p0, p1, p2, p3],  # bottom
                [p4, p5, p6, p7],  # top
                [p0, p1, p5, p4],
                [p1, p2, p6, p5],
                [p2, p3, p7, p6],
                [p3, p0, p4, p7],
            ]

            self.shelf_poly_verts.append(verts)

        self.objects = []
        for i in range(3):
            self.objects.append({
                "pos": self.floor_positions[i].copy(),
                "attached": False,
                "done": False,
                "target": self.shelf_positions[i].copy(),  # one shelf per object
            })

        # -------------------------------------------------
        # RTSS5 animation & IK validation state
        # -------------------------------------------------
        self.q_home = np.array([0.0, 0.20, 0.0, 0.0, 0.0], dtype=float)  # base yaw, lift, shoulder, elbow, wrist
        self.q = self.q_home.copy()                         # current joint config

        # Animation flags
        self.is_animating = False
        self.is_paused = False
        self.animation_sequence = []
        self.current_animation_step = 0
        self.total_interp_steps = 40   # interpolation steps between keyframes

        # For DH / FK-IK info box
        self.last_target_pos = None
        self.last_q_analytic = None
        self.last_q_numeric = None
        self.last_err_analytic = None
        self.last_err_numeric = None
        self.last_ok_analytic = None
        self.last_ok_numeric = None

        self.current_object_idx = 0
        self.carrying_object = None

        # Matplotlib setup
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("RTSS-5 Shelf Stacking Simulation")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")

        # Limits (adjust as needed)
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(0.0, 1.2)

        # -------------------------------------------------
        # Small floor table graphic vertices (built once)
        # -------------------------------------------------
        dx = self.table_width / 2.0
        dy = self.table_depth / 2.0

        x0, y0, z0 = self.table_center
        z1 = z0 + self.table_height

        p0 = [x0 - dx, y0 - dy, z0]
        p1 = [x0 + dx, y0 - dy, z0]
        p2 = [x0 + dx, y0 + dy, z0]
        p3 = [x0 - dx, y0 + dy, z0]

        p4 = [x0 - dx, y0 - dy, z1]
        p5 = [x0 + dx, y0 - dy, z1]
        p6 = [x0 + dx, y0 + dy, z1]
        p7 = [x0 - dx, y0 + dy, z1]

        self.table_verts = [
            [p0, p1, p2, p3],  # bottom
            [p4, p5, p6, p7],  # top
            [p0, p1, p5, p4],
            [p1, p2, p6, p5],
            [p2, p3, p7, p6],
            [p3, p0, p4, p7],
        ]

        # Path storage for EE trail
        self.ee_traj_x = []
        self.ee_traj_y = []
        self.ee_traj_z = []

        # Debug text placeholders so they persist across redraws
        self.text_goal_fk_str = ""
        self.text_ik_check_str = ""

        # Initial scene draw
        self.setup_scene()

        # Buttons
        self._add_buttons()

    def setup_scene(self):
        """Redraw robot, objects, shelves, and debug overlays for current state."""
        # Keep consistent axes after a clear
        self.ax.set_title("RTSS-5 Shelf Stacking Simulation")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(0.0, 1.2)

        # Draw table
        table = Poly3DCollection(self.table_verts, alpha=0.4, edgecolor="k")
        table.set_facecolor((0.7, 0.7, 0.7))
        self.ax.add_collection3d(table)

        # Draw shelves
        for verts in self.shelf_poly_verts:
            shelf_poly = Poly3DCollection(verts, alpha=0.6, edgecolor="k")
            shelf_poly.set_facecolor((0.85, 0.85, 0.85))
            self.ax.add_collection3d(shelf_poly)

        # Mark target markers (floor & shelves)
        self.ax.scatter(
            [p[0] for p in self.floor_positions],
            [p[1] for p in self.floor_positions],
            [p[2] for p in self.floor_positions],
            marker="s",
            s=40,
            label="Floor targets",
        )

        self.ax.scatter(
            [p[0] for p in self.shelf_positions],
            [p[1] for p in self.shelf_positions],
            [p[2] for p in self.shelf_positions],
            marker="^",
            s=40,
            label="Shelf targets",
        )

        # Object markers
        obj_x = [obj["pos"][0] for obj in self.objects]
        obj_y = [obj["pos"][1] for obj in self.objects]
        obj_z = [obj["pos"][2] for obj in self.objects]
        self.ax.scatter(
            obj_x, obj_y, obj_z,
            c=["red", "green", "blue"],
            s=60,
            label="Objects",
        )

        # Shelf target stars
        for i, pos in enumerate(self.shelf_positions):
            self.ax.plot(
                [pos[0]], [pos[1]], [pos[2]],
                marker="*", markersize=8,
                color="orange",
                linestyle="None",
                label="Shelf target" if i == 0 else None,
            )

        # Robot links
        joints, _ = self.robot.forward_kinematics(self.q)
        xs = [p[0] for p in joints]
        ys = [p[1] for p in joints]
        zs = [p[2] for p in joints]
        self.ax.plot(xs, ys, zs, "-o", lw=2.0)

        # EE trail
        ee = joints[-1]
        self.ee_traj_x.append(ee[0])
        self.ee_traj_y.append(ee[1])
        self.ee_traj_z.append(ee[2])
        self.ax.plot(self.ee_traj_x, self.ee_traj_y, self.ee_traj_z, ".", ms=2)

        self.ax.legend(loc="upper left")

        # Debug / legend text boxes for IK vs FK positions
        self.text_goal_fk = self.ax.text2D(
            0.8, 0.5, self.text_goal_fk_str, transform=self.ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7)
        )

        self.text_ik_check = self.ax.text2D(
            0.8, 1.0, self.text_ik_check_str, transform=self.ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7)
        )

        # DH / IK validation info box
        self.show_rtss5_dh_and_validation_box()
        

    # -----------------------------------------------------
    # Matplotlib GUI helpers
    # -----------------------------------------------------
    def _add_buttons(self):
        # Button for full automatic sequence
        ax_run = self.fig.add_axes([0.15, 0.02, 0.2, 0.06])
        self.btn_run = Button(ax_run, "Run full sequence")
        self.btn_run.on_clicked(self.on_run_full_sequence)

        # Button for workspace plot
        ax_ws = self.fig.add_axes([0.40, 0.02, 0.2, 0.06])
        self.btn_ws = Button(ax_ws, "Show workspace")
        self.btn_ws.on_clicked(self.on_show_workspace)

        # Button to reset to home
        ax_reset = self.fig.add_axes([0.65, 0.02, 0.2, 0.06])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self.on_reset)

        # Example: add simple start / pause buttons
        start_ax = self.fig.add_axes([0.80, 0.90, 0.08, 0.04])
        self.start_btn = Button(start_ax, "▶ Auto", color="lightgreen")
        self.start_btn.on_clicked(self.start_rtss5_animation)

        pause_ax = self.fig.add_axes([0.80, 0.85, 0.08, 0.04])
        self.pause_btn = Button(pause_ax, "▌ Pause", color="lightblue")
        self.pause_btn.on_clicked(self.toggle_pause)

    # -----------------------------------------------------
    # IK / FK debug legend update
    # -----------------------------------------------------
    def update_ik_legends(self,
                          target_pos: np.ndarray,
                          q_sol: np.ndarray,
                          success: bool):
        """
        Update the two text boxes showing:
        - Goal position and FK(q_sol)
        - IK status and position error
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        q_sol = np.asarray(q_sol, dtype=float).flatten()

        # Forward kinematics from the solution
        ee_pos = self.robot.end_effector_position(q_sol)
        error_vec = target_pos - ee_pos
        err_norm = float(np.linalg.norm(error_vec))

        # Left box: goal and FK position
        left_text = (
            "Goal position (m):\n"
            f"x: {target_pos[0]: .3f}\n"
            f"y: {target_pos[1]: .3f}\n"
            f"z: {target_pos[2]: .3f}\n"
            "\n"
            "FK(q) (m):\n"
            f"x: {ee_pos[0]: .3f}\n"
            f"y: {ee_pos[1]: .3f}\n"
            f"z: {ee_pos[2]: .3f}"
        )
        self.text_goal_fk_str = left_text

        # Right box: IK check
        right_text = (
            "IK check:\n"
            f"success: {success}\n"
            f"‖goal - FK(q)‖: {err_norm: .4f} m"
        )
        self.text_ik_check_str = right_text

        # Update text artists if they exist (after a redraw they are recreated)
        if hasattr(self, "text_goal_fk"):
            self.text_goal_fk.set_text(left_text)
        if hasattr(self, "text_ik_check"):
            self.text_ik_check.set_text(right_text)

    # -----------------------------------------------------
    # IK helper: chooses analytic or numerical method
    # -----------------------------------------------------
    def solve_ik(self, target_pos: np.ndarray, elbow_up: bool = True):
        """
        Solve IK for a given target position using either:
        - analytic IK (robot.ik_position), or
        - numerical IK (NumericalIKSolver), depending on self.use_numerical_ik.
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)

        if self.use_numerical_ik:
            # Use current configuration as initial guess for continuity
            q_init = self.q.copy()
            q_sol, success = self.ik_solver.solve(
                target_pos=target_pos,
                target_rot=None,    # position-only mode
                q_init=q_init,
            )
        else:
            # Analytic IK of RTSS5Robot (position-only)
            # desired_yaw=None, elbow_up flag is passed on
            q_sol, success = self.robot.ik_position(
                target_pos,
                desired_yaw=None,
                elbow_up=elbow_up,
            )

        # Update debug legends with the result
        self.update_ik_legends(target_pos, q_sol, success)
        return q_sol, success

    def solve_ik_rtss5(self, target_pos: np.ndarray):
        """
        Solve IK to a target position using:
          - Analytic IK from RTSS5Robot.ik_position
          - Numerical IK from NumericalIKSolver (DLS)
        Store both + errors for the DH/validation info box.

        Returns:
            q_use: joint vector to actually use (prefers analytic if valid).
        """
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)
        robot = self.robot

        # ---- Analytic IK ----
        q_analytic, ok_a = robot.ik_position(target_pos, desired_yaw=None, elbow_up=True)
        ee_a = robot.end_effector_position(q_analytic)
        err_a = np.linalg.norm(ee_a - target_pos)

        # ---- Numerical IK ----
        ik_solver = NumericalIKSolver(
            robot,
            max_iters=100,
            tol_pos=1e-3,
            tol_step=1e-4,
            damping=1e-3,
            position_only=True,
        )

        # Use analytic as good initial guess (or home if analytic fails badly)
        if ok_a:
            q_init = q_analytic.copy()
        else:
            q_init = self.q_home.copy()

        q_numeric, ok_n = ik_solver.solve(
            target_pos=target_pos,
            target_rot=None,
            q_init=q_init,
        )
        ee_n = robot.end_effector_position(q_numeric)
        err_n = np.linalg.norm(ee_n - target_pos)

        # ---- Store for info box ----
        self.last_target_pos = target_pos
        self.last_q_analytic = q_analytic
        self.last_q_numeric = q_numeric
        self.last_err_analytic = err_a
        self.last_err_numeric = err_n
        self.last_ok_analytic = ok_a
        self.last_ok_numeric = ok_n

        # ---- Decide what to use for motion ----
        if ok_a and err_a < 5e-3:
            return q_analytic
        elif ok_n and err_n < 5e-3:
            return q_numeric
        else:
            # Fallback: keep current config (or home)
            print("[WARN] IK failed or large error. Keeping home pose.")
            return self.q_home.copy()


    # -----------------------------------------------------
    # Core robot animation utilities
    # -----------------------------------------------------
    def update_robot_plot(self):
        """Refresh the Matplotlib scene based on the current robot/object state."""
        self.ax.clear()
        self.setup_scene()
        plt.draw()
        plt.pause(0.001)

    def move_joint_trajectory(self, q_start, q_end, steps=60):
        q_start = np.asarray(q_start, dtype=float).reshape(5)
        q_end = np.asarray(q_end, dtype=float).reshape(5)
        for alpha in np.linspace(0.0, 1.0, steps):
            self.q = (1 - alpha) * q_start + alpha * q_end
            # Keep within limits
            self.q = self.robot.clamp(self.q)
            # If any object is attached, move it with EE
            self._update_attached_objects()
            self.update_robot_plot()

    def _update_attached_objects(self):
        joints, _ = self.robot.forward_kinematics(self.q)
        ee_pos = joints[-1]
        for obj in self.objects:
            if obj["attached"]:
                obj["pos"] = ee_pos.copy()

    def create_keyframes_for_object(self, obj_idx: int):
        """
        Create a list of joint-space keyframes for object obj_idx:
          - Move from home -> above floor object
          - Move down to pick
          - Lift up
          - Move above shelf
          - Move down to place
          - Lift back up
        Each phase becomes a keyframe (joint vector) produced by IK.
        """
        obj = self.objects[obj_idx]
        floor_pos = obj["pos"].copy()
        shelf_pos = obj["target"].copy()

        # Slight safety offsets
        approach_height = 0.08
        place_height = 0.06
        hover_height = 0.10

        # ---- Cartesian waypoints (EE positions) ----
        # Home: we just use self.q_home, so no target pos needed
        p_home = self.robot.end_effector_position(self.q_home)

        # Above floor object
        p_above_floor = floor_pos.copy()
        p_above_floor[2] += approach_height

        # At floor object (for pick)
        p_pick = floor_pos.copy()

        # Lifted with object
        p_lift = p_above_floor.copy()

        # Above shelf
        p_above_shelf = shelf_pos.copy()
        p_above_shelf[2] += hover_height

        # At shelf (place)
        p_place = shelf_pos.copy()
        p_place[2] += place_height

        # Retreat above shelf again
        p_retreat = p_above_shelf.copy()

        # ---- Solve IK for each waypoint ----
        keyframes = []

        # 0) Home
        q0 = self.q_home.copy()
        keyframes.append({"q": q0, "action": "move", "crate_attached": False, "crate_on_shelf": False})

        # 1) Above floor
        q1 = self.solve_ik_rtss5(p_above_floor)
        keyframes.append({"q": q1, "action": "move", "crate_attached": False, "crate_on_shelf": False})

        # 2) Pick
        q2 = self.solve_ik_rtss5(p_pick)
        keyframes.append({"q": q2, "action": "attach", "crate_attached": True, "crate_on_shelf": False})

        # 3) Lift
        q3 = self.solve_ik_rtss5(p_lift)
        keyframes.append({"q": q3, "action": "move_with_object", "crate_attached": True, "crate_on_shelf": False})

        # 4) Above shelf
        q4 = self.solve_ik_rtss5(p_above_shelf)
        keyframes.append({"q": q4, "action": "move_with_object", "crate_attached": True, "crate_on_shelf": False})

        # 5) Place
        q5 = self.solve_ik_rtss5(p_place)
        keyframes.append({"q": q5, "action": "place", "crate_attached": False, "crate_on_shelf": True})

        # 6) Retreat
        q6 = self.solve_ik_rtss5(p_retreat)
        keyframes.append({"q": q6, "action": "move", "crate_attached": False, "crate_on_shelf": True})

        return keyframes

    def interpolate_joints(self, q_start: np.ndarray, q_end: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation in joint space."""
        return q_start + t * (q_end - q_start)

    def generate_animation_sequence_for_object(self, obj_idx: int):
        """
        Use keyframes (joint vectors + actions) and build a dense
        animation sequence with interpolated joint states.
        """
        keyframes = self.create_keyframes_for_object(obj_idx)
        sequence = []

        for k in range(len(keyframes) - 1):
            kf_start = keyframes[k]
            kf_end = keyframes[k + 1]
            q_start = kf_start["q"]
            q_end = kf_end["q"]

            # interpolate between keyframes
            for i in range(self.total_interp_steps):
                t = i / (self.total_interp_steps - 1)
                q_interp = self.interpolate_joints(q_start, q_end, t)

                if kf_end["action"] == "attach":
                    action = "move"
                elif kf_end["action"] == "place":
                    action = "move_with_object"
                else:
                    action = kf_end["action"]

                sequence.append({
                    "joints": q_interp,
                    "action": action,
                    "obj_idx": obj_idx,
                })

            # Insert discrete attach / place steps explicitly
            if kf_end["action"] == "attach":
                sequence.append({
                    "joints": q_end,
                    "action": "attach",
                    "obj_idx": obj_idx,
                })
            elif kf_end["action"] == "place":
                sequence.append({
                    "joints": q_end,
                    "action": "place",
                    "obj_idx": obj_idx,
                })

        return sequence

    def update_animation(self, frame):
        """
        Called by FuncAnimation: apply next step of animation_sequence.
        """
        if self.is_paused or not self.is_animating:
            return []

        if self.current_animation_step >= len(self.animation_sequence):
            # Finished current object, try next one
            self.current_animation_step = 0
            self.current_object_idx += 1

            if self.current_object_idx >= len(self.objects):
                # All objects done
                self.is_animating = False
                self.current_object_idx = 0
                return []

            # generate sequence for next object
            self.animation_sequence = self.generate_animation_sequence_for_object(self.current_object_idx)
            return []

        step = self.animation_sequence[self.current_animation_step]
        self.q = step["joints"]
        obj_idx = step["obj_idx"]
        action = step["action"]

        # Update object states (similar idea to your PDE4431 code)
        obj = self.objects[obj_idx]
        if action == "attach":
            obj["attached"] = True
            obj["done"] = False
            self.carrying_object = obj_idx
        elif action == "place":
            obj["attached"] = False
            obj["done"] = True
            self.carrying_object = None
            # snap object to its shelf target
            obj["pos"] = obj["target"].copy()
        elif action == "move_with_object":
            obj["attached"] = True
            obj["done"] = False
            self.carrying_object = obj_idx
            # move object with EE (approx)
            ee_pos = self.robot.end_effector_position(self.q)
            obj["pos"] = ee_pos.copy()

        # Redraw scene
        self.ax.clear()
        self.setup_scene()   # make sure setup_scene uses self.q to draw robot pose
        self.current_animation_step += 1
        return []

    def start_rtss5_animation(self, event=None):
        if self.is_animating:
            return
        self.is_animating = True
        self.is_paused = False
        self.current_object_idx = 0
        self.current_animation_step = 0
        self.animation_sequence = self.generate_animation_sequence_for_object(0)

        self.anim = animation.FuncAnimation(
            self.fig,
            self.update_animation,
            frames=len(self.animation_sequence) * len(self.objects),
            interval=50,
            repeat=False,
            blit=False,
        )

    def toggle_pause(self, event=None):
        if not self.is_animating:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.label.set_text("▶ Resume")
            if hasattr(self, "anim"):
                self.anim.event_source.stop()
        else:
            self.pause_btn.label.set_text("▌ Pause")
            if hasattr(self, "anim"):
                self.anim.event_source.start()
        self.fig.canvas.draw_idle()

    def show_rtss5_dh_and_validation_box(self):
        """
        Display RTSS5 DH parameters and the last FK/IK validation info
        as a text box overlay in the main simulation.
        """
        q = self.q.copy()
        q1, q2, q3, q4, q5 = q
        r = self.robot

        # Standard DH model we defined earlier:
        # J1: a=0,    alpha=0, d=h0,   theta=q1
        # J2: a=0,    alpha=0, d=q2,   theta=0
        # J3: a=L1,   alpha=0, d=0,    theta=q3
        # J4: a=L2,   alpha=0, d=0,    theta=q4
        # J5: a=Ltool,alpha=0, d=0,    theta=q5

        dh_rows = [
            ("J1", 0.0,            0.0,           r.h0,         np.degrees(q1)),
            ("J2", 0.0,            0.0,           q2,           0.0),
            ("J3", r.L1,           0.0,           0.0,          np.degrees(q3)),
            ("J4", r.L2,           0.0,           0.0,          np.degrees(q4)),
            ("J5", r.L_tool,       0.0,           0.0,          np.degrees(q5)),
        ]

        txt = "RTSS-5 DENAVIT–HARTENBERG TABLE\n"
        txt += "═" * 55 + "\n"
        txt += " Joint |   a (m)  |   α (°)  |   d (m)  |   θ (°)  \n"
        txt += "───────┼──────────┼──────────┼──────────┼──────────\n"

        for name, a, alpha, d, theta in dh_rows:
            txt += f"  {name:4} |  {a:5.3f}   | {alpha:7.1f}  | {d:6.3f}  | {theta:7.1f}\n"

        txt += "\nFK/IK VALIDATION (last solve_ik_rtss5):\n"
        if self.last_target_pos is not None:
            tp = self.last_target_pos
            txt += f"  Target p* = [{tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}] m\n"
            txt += f"  Analytic OK: {self.last_ok_analytic},  ‖p* - FK(q_a)‖ = {self.last_err_analytic:.3e} m\n"
            txt += f"  Numeric  OK: {self.last_ok_numeric},  ‖p* - FK(q_n)‖ = {self.last_err_numeric:.3e} m\n"
        else:
            txt += "  (No IK solve performed yet)\n"

        self.ax.text2D(
            0.02, 0.02,
            txt,
            transform=self.ax.transAxes,
            fontsize=8.0,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="mistyrose",
                alpha=0.95,
                edgecolor="darkred",
                linewidth=1.5,
            ),
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    # -----------------------------------------------------
    # Pick & place logic
    # -----------------------------------------------------
    def pick_and_place_object(self, idx: int):
        """
        Perform pick & place for object at index idx:
        floor_positions[idx] -> shelf_positions[idx]
        """
        floor_pos = self.floor_positions[idx]
        shelf_pos = self.objects[idx]["target"]

        # Pre-pick: move above floor target (or table target)
        pre_pick = floor_pos.copy()
        pre_pick[2] += 0.10

        # Pre-place: move above shelf target
        pre_place = shelf_pos.copy()
        pre_place[2] += 0.10

        # 1) Move from current pose to pre-pick
        q_pre_pick, ok1 = self.solve_ik(pre_pick, elbow_up=True)
        if not ok1:
            print(f"[WARN] Pre-pick pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q, q_pre_pick, steps=60)

        # 2) Move down to pick
        q_pick, ok2 = self.solve_ik(floor_pos, elbow_up=True)
        if not ok2:
            print(f"[WARN] Pick pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q, q_pick, steps=40)

        # Attach object
        self.objects[idx]["attached"] = True
        self._update_attached_objects()

        # 3) Lift back to pre-pick
        q_pre_pick_up, ok3 = self.solve_ik(pre_pick, elbow_up=True)
        if not ok3:
            print(f"[WARN] Pre-pick-up pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q, q_pre_pick_up, steps=40)

        # 4) Move to pre-place
        q_pre_place, ok4 = self.solve_ik(pre_place, elbow_up=True)
        if not ok4:
            print(f"[WARN] Pre-place pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q, q_pre_place, steps=60)

        # 5) Move down to place
        q_place, ok5 = self.solve_ik(shelf_pos, elbow_up=True)
        if not ok5:
            print(f"[WARN] Place pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q, q_place, steps=40)

        # Detach object at shelf
        self.objects[idx]["attached"] = False
        self.objects[idx]["pos"] = shelf_pos.copy()
        self.objects[idx]["done"] = True

        # 6) Return to pre-place, then home
        q_pre_place_exit, ok6 = self.solve_ik(pre_place, elbow_up=True)
        if ok6:
            self.move_joint_trajectory(self.q, q_pre_place_exit, steps=40)

        self.move_joint_trajectory(self.q, self.q_home, steps=60)

    # -----------------------------------------------------
    # Button callbacks
    # -----------------------------------------------------
    def on_run_full_sequence(self, _event):
        """Run pick & place for all three objects sequentially."""
        print("[INFO] Running full pick & place sequence...")
        for idx in range(3):
            if not self.objects[idx]["done"]:
                self.pick_and_place_object(idx)

    def on_show_workspace(self, _event):
        """Show a scatter of sampled workspace points."""
        print("[INFO] Sampling workspace...")
        pts = self.robot.sample_workspace(num_samples=800)
        self.ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=1, alpha=0.2, label="Workspace samples",
        )
        plt.draw()

    def on_reset(self, _event):
        """Reset robot and objects to initial state."""
        print("[INFO] Resetting simulation...")
        self.q = self.q_home.copy()
        self.ee_traj_x.clear()
        self.ee_traj_y.clear()
        self.ee_traj_z.clear()
        self.animation_sequence = []
        self.current_animation_step = 0
        self.current_object_idx = 0
        self.is_animating = False
        self.is_paused = False
        self.carrying_object = None

        for i, obj in enumerate(self.objects):
            obj["attached"] = False
            obj["done"] = False
            obj["pos"] = self.floor_positions[i].copy()

        self.update_robot_plot()

    # -----------------------------------------------------
    # Public entrypoint
    # -----------------------------------------------------
    def run(self):
        """Start the Matplotlib event loop."""
        self.update_robot_plot()
        plt.show()

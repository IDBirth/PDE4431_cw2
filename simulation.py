import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from robot import RTSS5Robot



class ShelfStackingSimulation:
    """
    Simulation of the RTSS-5 robot performing pick & place:
    - Three objects on the floor (same radial distance, slightly different angles)
    - Three shelf positions at increasing heights, same radial distance.
    """

    def __init__(self):
        self.robot = RTSS5Robot()

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

        # Shelves: unchanged (still at radius R_shelf in front)
        self.z_shelves = [0.30, 0.55, 0.80]
        self.shelf_positions = []
        for i in range(3):
            angle = self.theta_base  # aligned shelves
            x = self.R_shelf * np.cos(angle)
            y = self.R_shelf * np.sin(angle)
            z = self.z_shelves[i]
            self.shelf_positions.append(np.array([x, y, z]))

        # Object states: position + flags
        self.objects = []
        for i in range(3):
            self.objects.append({
                "pos": self.floor_positions[i].copy(),
                "attached": False,
                "done": False,
            })

        # Robot initial joint configuration (home)
        self.q_home = np.array([0.0, 0.40, 0.0, 0.0, 0.0], dtype=float)
        self.q_current = self.q_home.copy()

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

        # Plot elements for robot
        joints, _ = self.robot.forward_kinematics(self.q_current)
        xs = [p[0] for p in joints]
        ys = [p[1] for p in joints]
        zs = [p[2] for p in joints]

        (self.link_line,) = self.ax.plot(xs, ys, zs, "-o", lw=2.0)
        self.ee_path, = self.ax.plot([], [], [], ".", ms=2)  # trail

        # Plot target markers (floor & shelves)
        self.floor_scatter = self.ax.scatter(
            [p[0] for p in self.floor_positions],
            [p[1] for p in self.floor_positions],
            [p[2] for p in self.floor_positions],
            marker="s",
            s=40,
            label="Floor targets",
        )

        self.shelf_scatter = self.ax.scatter(
            [p[0] for p in self.shelf_positions],
            [p[1] for p in self.shelf_positions],
            [p[2] for p in self.shelf_positions],
            marker="^",
            s=40,
            label="Shelf targets",
        )

        # Object markers (start on floor)
        obj_x = [obj["pos"][0] for obj in self.objects]
        obj_y = [obj["pos"][1] for obj in self.objects]
        obj_z = [obj["pos"][2] for obj in self.objects]
        self.obj_scatter = self.ax.scatter(
            obj_x, obj_y, obj_z,
            c=["red", "green", "blue"],
            s=60,
            label="Objects",
        )

        # -------------------------------------------------
        # Small floor table graphic
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

        table_verts = [
            [p0, p1, p2, p3],  # bottom
            [p4, p5, p6, p7],  # top
            [p0, p1, p5, p4],
            [p1, p2, p6, p5],
            [p2, p3, p7, p6],
            [p3, p0, p4, p7],
        ]

        self.table = Poly3DCollection(table_verts, alpha=0.4, edgecolor="k")
        self.table.set_facecolor((0.7, 0.7, 0.7))
        self.ax.add_collection3d(self.table)

        #add legend
        self.ax.legend(loc="upper left")

        # Path storage for EE trail
        self.ee_traj_x = []
        self.ee_traj_y = []
        self.ee_traj_z = []

        # Buttons
        self._add_buttons()








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

    # -----------------------------------------------------
    # Core robot animation utilities
    # -----------------------------------------------------
    def update_robot_plot(self):
        joints, _ = self.robot.forward_kinematics(self.q_current)
        xs = [p[0] for p in joints]
        ys = [p[1] for p in joints]
        zs = [p[2] for p in joints]

        self.link_line.set_data(xs, ys)
        self.link_line.set_3d_properties(zs)

        ee = joints[-1]
        self.ee_traj_x.append(ee[0])
        self.ee_traj_y.append(ee[1])
        self.ee_traj_z.append(ee[2])

        self.ee_path.set_data(self.ee_traj_x, self.ee_traj_y)
        self.ee_path.set_3d_properties(self.ee_traj_z)

        # Update object positions
        obj_x = [obj["pos"][0] for obj in self.objects]
        obj_y = [obj["pos"][1] for obj in self.objects]
        obj_z = [obj["pos"][2] for obj in self.objects]
        self.obj_scatter._offsets3d = (obj_x, obj_y, obj_z)

        plt.draw()
        plt.pause(0.001)

    def move_joint_trajectory(self, q_start, q_end, steps=60):
        q_start = np.asarray(q_start, dtype=float).reshape(5)
        q_end = np.asarray(q_end, dtype=float).reshape(5)
        for alpha in np.linspace(0.0, 1.0, steps):
            self.q_current = (1 - alpha) * q_start + alpha * q_end
            # Keep within limits
            self.q_current = self.robot.clamp(self.q_current)
            # If any object is attached, move it with EE
            self._update_attached_objects()
            self.update_robot_plot()

    def _update_attached_objects(self):
        joints, _ = self.robot.forward_kinematics(self.q_current)
        ee_pos = joints[-1]
        for obj in self.objects:
            if obj["attached"]:
                obj["pos"] = ee_pos.copy()

    # -----------------------------------------------------
    # Pick & place logic
    # -----------------------------------------------------
    def pick_and_place_object(self, idx: int):
        """
        Perform pick & place for object at index idx:
        floor_positions[idx] -> shelf_positions[idx]
        """
        floor_pos = self.floor_positions[idx]
        shelf_pos = self.shelf_positions[idx]

        # Pre-pick: move above floor target
        pre_pick = floor_pos.copy()
        pre_pick[2] += 0.10

        # Pre-place: move above shelf target
        pre_place = shelf_pos.copy()
        pre_place[2] += 0.10

        # 1) Move from home to pre-pick
        q_pre_pick, ok1 = self.robot.ik_position(pre_pick)
        if not ok1:
            print(f"[WARN] Pre-pick pose unreachable for object {idx}")
            return

        self.move_joint_trajectory(self.q_current, q_pre_pick, steps=60)

        # 2) Move down to pick
        q_pick, ok2 = self.robot.ik_position(floor_pos)
        if not ok2:
            print(f"[WARN] Pick pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q_current, q_pick, steps=40)

        # Attach object
        self.objects[idx]["attached"] = True
        self._update_attached_objects()

        # 3) Lift back to pre-pick
        self.move_joint_trajectory(self.q_current, q_pre_pick, steps=40)

        # 4) Move to pre-place
        q_pre_place, ok3 = self.robot.ik_position(pre_place)
        if not ok3:
            print(f"[WARN] Pre-place pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q_current, q_pre_place, steps=60)

        # 5) Move down to place
        q_place, ok4 = self.robot.ik_position(shelf_pos)
        if not ok4:
            print(f"[WARN] Place pose unreachable for object {idx}")
            return
        self.move_joint_trajectory(self.q_current, q_place, steps=40)

        # Detach object at shelf
        self.objects[idx]["attached"] = False
        self.objects[idx]["pos"] = shelf_pos.copy()
        self.objects[idx]["done"] = True

        # 6) Return to pre-place, then home
        self.move_joint_trajectory(self.q_current, q_pre_place, steps=40)
        self.move_joint_trajectory(self.q_current, self.q_home, steps=60)

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
        self.q_current = self.q_home.copy()
        self.ee_traj_x.clear()
        self.ee_traj_y.clear()
        self.ee_traj_z.clear()

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

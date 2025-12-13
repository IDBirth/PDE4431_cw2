import numpy as np
from math import atan2, sqrt, cos, sin, pi, degrees, radians
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used by mpl_toolkits)
from matplotlib.widgets import Slider, TextBox, Button
import roboticstoolbox as rtb
from spatialmath import SE3
import sys


class RobotArm:
    """
    A 4-DOF R-P-R-R robotic arm for PDE4431 CW2:
    - Includes one prismatic joint
    - Uses DH parameters
    - Supports FK, numerical IK, 3D visualisation
    - Demonstrates a 3-block stacking task (floor -> 3 shelves)
    """

    def __init__(self):
        # ---- DH parameters: [theta, d, a, alpha, joint_type] ----
        # Units: meters and radians
        # Joint 1: Revolute base
        # Joint 2: Prismatic vertical (z)
        # Joint 3: Revolute reach
        # Joint 4: Revolute wrist/reach
        self.dh_params = [
            [0.0, 0.0, 0.0, 0.0, 'R'],   # Joint 1: base rotation about z
            [0.0, 0.0, 0.0, 0.0, 'P'],   # Joint 2: vertical prismatic along z
            [0.0, 0.0, 0.35, 0.0, 'R'],  # Joint 3: horizontal link (reach)
            [0.0, 0.0, 0.25, 0.0, 'R'],  # Joint 4: horizontal link (wrist / reach)
        ]

        # Joint limits
        # - Revolute joints: degrees
        # - Prismatic joint (Joint 2): meters of extension
        self.joint_limits = [
            [-180, 180],   # Joint 1 (deg)
            [0.05, 0.80],  # Joint 2 (m)  height range (z = q2)
            [-120, 120],   # Joint 3 (deg)
            [-150, 150],   # Joint 4 (deg)
        ]

        # Names & DH component lists
        self.joint = [f'Joint{i+1}' for i in range(len(self.dh_params))]
        self.theta = [p[0] for p in self.dh_params]
        self.d = [p[1] for p in self.dh_params]
        self.a = [p[2] for p in self.dh_params]
        self.alpha = [p[3] for p in self.dh_params]
        self.joint_types = [p[4] for p in self.dh_params]  # 'R' or 'P'

        # Joint variables: radians for R, meters for P
        self.joint_variables = np.zeros(len(self.dh_params))
        # A comfortable "home" pose (slightly up)
        self.joint_variables[1] = 0.3   # q2 = 0.3 m
        self.q_home = self.joint_variables.copy()

        # Storage for kinematic states
        n = len(self.dh_params)
        self.joint_positions = [np.zeros(3) for _ in range(n)]
        self.joint_orientations = [np.zeros(3) for _ in range(n)]
        self.joint_states = [np.zeros(6) for _ in range(n)]
        self.joint_rotation_matrices = [np.zeros((3, 3)) for _ in range(n)]
        self.end_effector_rotation_matrix = np.zeros((3, 3))
        self.end_effector_position = np.zeros(3)
        self.end_effector_orientation = np.zeros(3)
        self.end_effector_state = np.zeros(6)
        self.T_0_6 = np.eye(4)
        self.sliders = []
        self.bounding_box = []
        self.info_text_annotation = None
        self.workspace_plots = []

        # ---- Define task positions: 3 blocks on floor → 3 shelves ----
        floor_z = 0.05  # "floor" height the EE will use
        self.floor_targets = [
            np.array([0.50,  0.00, floor_z, 0.0, 0.0, 0.0]),
            np.array([0.40,  0.15, floor_z, 0.0, 0.0, 0.0]),
            np.array([0.40, -0.15, floor_z, 0.0, 0.0, 0.0]),
        ]

        self.shelf_targets = [
            np.array([0.50,  0.00, 0.30, 0.0, 0.0, 0.0]),  # low shelf
            np.array([0.55,  0.15, 0.50, 0.0, 0.0, 0.0]),  # mid shelf
            np.array([0.55, -0.15, 0.70, 0.0, 0.0, 0.0]),  # high shelf
        ]

        # ---- Build DHRobot model (Robotics Toolbox) ----
        self.links = []
        for (theta, d, a, alpha, jtype) in self.dh_params:
            if jtype == 'R':
                link = rtb.RevoluteDH(d=d, a=a, alpha=alpha, offset=theta)
            else:  # 'P'
                link = rtb.PrismaticDH(theta=theta, a=a, alpha=alpha, offset=d)
            self.links.append(link)

        self.robot = rtb.DHRobot(self.links, name="BlockStacker")

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    def homogeneous_transform_matrix(self, theta, d, a, alpha, joint_type, joint_var):
        """
        DH homogeneous transform for either a revolute or prismatic joint.

        - If joint_type == 'R': joint_var is an angle (rad), added to theta.
        - If joint_type == 'P': joint_var is a displacement (m), added to d.
        """
        if joint_type == 'R':
            th = theta + joint_var
            dd = d
        else:  # 'P'
            th = theta
            dd = d + joint_var

        T = np.array([
            [cos(th), -sin(th) * cos(alpha),  sin(th) * sin(alpha), a * cos(th)],
            [sin(th),  cos(th) * cos(alpha), -cos(th) * sin(alpha), a * sin(th)],
            [0,        sin(alpha),           cos(alpha),            dd],
            [0,        0,                    0,                     1]
        ])
        return T

    def frame_orientation(self, rotation_matrix):
        """
        Calculate fixed frame Euler angles (alpha, beta, gamma) from a rotation matrix.
        """
        r11 = rotation_matrix[0, 0]
        r12 = rotation_matrix[0, 1]
        r13 = rotation_matrix[0, 2]
        r21 = rotation_matrix[1, 0]
        r22 = rotation_matrix[1, 1]
        r23 = rotation_matrix[1, 2]
        r31 = rotation_matrix[2, 0]
        r32 = rotation_matrix[2, 1]
        r33 = rotation_matrix[2, 2]

        beta = atan2(-r31, sqrt(r11**2 + r21**2))
        if cos(beta) != 0:
            alpha = atan2(r21 / cos(beta), r11 / cos(beta))
            gamma = atan2(r32 / cos(beta), r33 / cos(beta))
        elif beta == pi / 2:
            alpha = 0
            gamma = atan2(r12, r22)
        elif beta == -pi / 2:
            alpha = 0
            gamma = -atan2(r12, r22)

        return np.array([alpha, beta, gamma])

    def fixed_frame_transformation_matrix(self, desired_state):
        """
        Generates a transformation matrix given a desired translation and rotation.

        desired_state: [x, y, z, alpha, beta, gamma]
        All in meters/radians.
        """
        px, py, pz, alpha, beta, gamma = desired_state

        T = np.array([
            [cos(alpha) * cos(beta),
             cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
             cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
             px],
            [sin(alpha) * cos(beta),
             sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
             sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
             py],
            [-sin(beta),
             cos(beta) * sin(gamma),
             cos(beta) * cos(gamma),
             pz],
            [0, 0, 0, 1]
        ])
        return T

    def forward_kinematics(self, joint_variables=None, print=False):
        """
        Calculate end-effector pose given joint variables.
        joint_variables: radians for revolute joints, meters for prismatic.
        """
        if joint_variables is None:
            joint_variables = self.joint_variables
        else:
            self.joint_variables = np.array(joint_variables)

        self.T_0_6 = np.eye(4)

        for i in range(len(self.dh_params)):
            T_local = self.homogeneous_transform_matrix(
                self.theta[i],
                self.d[i],
                self.a[i],
                self.alpha[i],
                self.joint_types[i],
                self.joint_variables[i]
            )
            self.T_0_6 = self.T_0_6 @ T_local
            self.joint_positions[i] = self.T_0_6[:3, 3]
            self.joint_rotation_matrices[i] = self.T_0_6[:3, :3]
            self.joint_orientations[i] = self.frame_orientation(self.T_0_6[:3, :3])
            self.joint_states[i] = np.concatenate(
                [self.joint_positions[i], self.joint_orientations[i]]
            )

        self.end_effector_rotation_matrix = self.T_0_6[:3, :3]
        self.end_effector_position = self.joint_positions[-1]
        self.end_effector_orientation = self.joint_orientations[-1]
        self.end_effector_state = self.joint_states[-1]

        if print:
            print(
                f'Joint Variables: {self.joint_variables}\n'
                f'End Effector Position: '
                f'({self.end_effector_position[0]:.3f}, '
                f'{self.end_effector_position[1]:.3f}, '
                f'{self.end_effector_position[2]:.3f}) m\n'
                f'End Effector Orientation (deg): '
                f'({degrees(self.end_effector_orientation[0]):.2f}, '
                f'{degrees(self.end_effector_orientation[1]):.2f}, '
                f'{degrees(self.end_effector_orientation[2]):.2f})'
            )
            print("-" * 50)

        return self.end_effector_state

    def inverse_kinematics(self, desired_state):
        """
        Numerical IK using Robotics Toolbox (ikine_LM).
        desired_state: [x, y, z, alpha, beta, gamma] (angles in radians)
        We only constrain position (x, y, z); orientation is free.
        """
        T_des = SE3(self.fixed_frame_transformation_matrix(desired_state))

        # mask: [x, y, z, Rx, Ry, Rz] -> track only position
        sol = self.robot.ikine_LM(T_des, mask=[1, 1, 1, 0, 0, 0])

        if not sol.success:
            print("IK failed for target:", desired_state)
            return None

        q = sol.q  # numpy array (len = 4)

        # Simple joint-limit check
        for i, val in enumerate(q):
            if self.joint_types[i] == 'R':
                angle_deg = degrees(val)
                if angle_deg < self.joint_limits[i][0] or angle_deg > self.joint_limits[i][1]:
                    print(f"Solution violates limits of {self.joint[i]} (deg): {angle_deg:.1f}")
            else:  # prismatic
                if val < self.joint_limits[i][0] or val > self.joint_limits[i][1]:
                    print(f"Solution violates limits of {self.joint[i]} (m): {val:.3f}")
        return q

    # ------------------------------------------------------------------
    # Visualisation and GUI
    # ------------------------------------------------------------------

    def setup_visualization(self):
        """
        Set up the 3D visualization environment with interactive controls.
        - Sliders and textboxes for each joint
        - Buttons for workspace and block stacking
        """
        self.env = rtb.backends.PyPlot.PyPlot()
        self.env.launch()

        # Adjust axes position to leave space for widgets
        self.env.ax.set_position([0.1, 0.2, 0.8, 0.8])
        self.ax = self.env.ax

        # Draw a simple base
        base_vertices = np.array([
            [-0.07, -0.07, 0],
            [-0.07,  0.07, 0],
            [ 0.07,  0.07, 0],
            [ 0.07, -0.07, 0],
            [-0.07, -0.07, -0.3],
            [-0.07,  0.07, -0.3],
            [ 0.07,  0.07, -0.3],
            [ 0.07, -0.07, -0.3]
        ])

        faces = [
            [0, 1, 2], [0, 2, 3],   # Top
            [4, 5, 6], [4, 6, 7],   # Bottom
            [0, 1, 5], [0, 5, 4],   # Front
            [2, 3, 7], [2, 7, 6],   # Back
            [0, 3, 7], [0, 7, 4],   # Left
            [1, 2, 6], [1, 6, 5]    # Right
        ]

        self.ax.plot_trisurf(
            base_vertices[:, 0],
            base_vertices[:, 1],
            base_vertices[:, 2],
            triangles=faces,
            color='gray',
            shade=True,
            alpha=0.8
        )

        # Draw simple shelves (flat rectangles) at each shelf height
        for shelf in self.shelf_targets:
            x_c, y_c, z_c = shelf[:3]
            shelf_size_x = 0.2
            shelf_size_y = 0.1
            xs = [x_c - shelf_size_x / 2, x_c + shelf_size_x / 2]
            ys = [y_c - shelf_size_y / 2, y_c + shelf_size_y / 2]
            X, Y = np.meshgrid(xs, ys)
            Z = np.ones_like(X) * z_c
            self.ax.plot_surface(X, Y, Z, alpha=0.2)

        # Add robot to environment
        self.env.add(
            self.robot,
            jointaxes=False,
            eeframe=True,
            shadow=True,
            display=True
        )

        # EE annotation box
        self.ee_annotation = self.ax.figure.text(
            0.5,
            0.95,
            '',
            transform=self.ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round,pad=0.5',
                fc='white',
                alpha=0.8,
                edgecolor='red'
            )
        )

        # Axis labels
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('4-DOF R-P-R-R Block Stacking Robot')

        # Slider / textbox / button axes
        slider_axes = [plt.axes([0.25, 0.20 - i * 0.03, 0.4, 0.03])
                       for i in range(len(self.links))]
        textbox_axes = [plt.axes([0.1, 0.20 - i * 0.03, 0.1, 0.03])
                        for i in range(len(self.links))]
        button_axes = [plt.axes([0.14, 0.20 - i * 0.03, 0.075, 0.03])
                       for i in range(len(self.links))]

        # Workspace button
        workspace_button_ax = plt.axes([0.7, 0.85, 0.2, 0.04])
        self.workspace_button = Button(
            workspace_button_ax,
            'Draw Workspace',
            color='lightgray',
            hovercolor='0.7'
        )
        self.workspace_button.on_clicked(lambda event: self.plot_workspace())

        # Reset button
        reset_button_ax = plt.axes([0.7, 0.80, 0.2, 0.04])
        self.reset_button = Button(
            reset_button_ax,
            'Reset',
            color='lightgray',
            hovercolor='0.9'
        )
        self.reset_button.on_clicked(lambda event: self.reset_robot())

        # Run Block Stacking button
        run_button_ax = plt.axes([0.7, 0.75, 0.2, 0.04])
        self.run_button = Button(
            run_button_ax,
            'Run Block Stacking',
            color='lightgray',
            hovercolor='0.9'
        )
        self.run_button.on_clicked(lambda event: self.execute_block_stacking_sequence())

        # Sliders, textboxes, per-joint update buttons
        self.sliders = []
        self.textboxes = []
        self.buttons = []

        for i, (slider_ax, textbox_ax, button_ax) in enumerate(
            zip(slider_axes, textbox_axes, button_axes)
        ):
            if self.joint_types[i] == 'R':
                s_min, s_max = self.joint_limits[i]
                valinit = 0.0
                label = f'J{i+1} (deg):'
            else:  # 'P'
                s_min, s_max = self.joint_limits[i]
                valinit = 0.3
                label = f'J{i+1} (m):'

            slider = Slider(slider_ax, '', s_min, s_max, valinit=valinit)
            textbox = TextBox(textbox_ax, label, initial='', color='lightgray')
            button = Button(button_ax, 'Update', color='lightgray', hovercolor='0.9')

            self.sliders.append(slider)
            self.textboxes.append(textbox)
            self.buttons.append(button)

            slider.on_changed(lambda value: self.update_plot())
            textbox.on_text_change(lambda value, idx=i: self.update_textbox_value(value, idx))
            button.on_clicked(lambda event, idx=i: self.update_from_textbox(idx))

        # Start in home configuration and sync sliders
        self.set_joint_variables(self.q_home)

        try:
            plt.get_current_fig_manager().full_screen_toggle()
        except Exception:
            # Some backends don't support this; ignore.
            pass

    def update_textbox_value(self, value, index):
        """
        Parse textbox value.
        - Revolute: degrees
        - Prismatic: meters
        """
        try:
            val = float(value)
            low, high = self.joint_limits[index]

            if not (low <= val <= high):
                unit = 'deg' if self.joint_types[index] == 'R' else 'm'
                print(
                    f"Invalid input for Joint {index+1}. "
                    f"Enter between {low} and {high} {unit}."
                )
                return

            if self.joint_types[index] == 'R':
                self.joint_variables[index] = radians(val)
            else:
                self.joint_variables[index] = val

            self.textboxes[index].set_val(value)

        except ValueError:
            print(f"Invalid input for Joint {index+1}. Please enter a number.")

    def update_from_textbox(self, index):
        txt = self.textboxes[index].text
        try:
            val = float(txt)
        except ValueError:
            print(f"Textbox {index+1} is not a number.")
            return

        self.sliders[index].set_val(val)
        self.update_plot()

    def set_joint_variables(self, joint_variables):
        """
        joint_variables: list/array
        - Revolute: radians
        - Prismatic: meters
        """
        self.joint_variables = np.array(joint_variables)
        for i in range(len(self.sliders)):
            if self.joint_types[i] == 'R':
                self.sliders[i].set_val(degrees(self.joint_variables[i]))
            else:
                self.sliders[i].set_val(self.joint_variables[i])
        self.update_plot()

    def update_plot(self):
        new_vars = []
        for i, slider in enumerate(self.sliders):
            if self.joint_types[i] == 'R':
                new_vars.append(radians(slider.val))
            else:
                new_vars.append(slider.val)
        self.joint_variables = np.array(new_vars)
        self.forward_kinematics(self.joint_variables)
        self.plot_robot_arm()
        return

    def plot_robot_arm(self):
        # Create backend if not exists
        if not hasattr(self, 'env'):
            self.env = rtb.backends.PyPlot.PyPlot()
            self.env.launch()

        # Update robot configuration
        self.robot.q = self.joint_variables

        ee_text = (
            f'End Effector:\n'
            f' Position: ({self.end_effector_position[0]:.3f}, '
            f'{self.end_effector_position[1]:.3f}, '
            f'{self.end_effector_position[2]:.3f}) m\n'
            f' Orientation: ({degrees(self.end_effector_orientation[0]):.2f}, '
            f'{degrees(self.end_effector_orientation[1]):.2f}, '
            f'{degrees(self.end_effector_orientation[2]):.2f}) deg'
        )
        if hasattr(self, "ee_annotation") and self.ee_annotation is not None:
            self.ee_annotation.set_text(ee_text)

        self.env.step()
        return

    def reset_robot(self):
        # Remove all workspace plots
        for plot in self.workspace_plots:
            try:
                plot.remove()
            except Exception:
                pass
        self.workspace_plots = []

        self.set_joint_variables(self.q_home)
        return

    # ------------------------------------------------------------------
    # Workspace visualisation
    # ------------------------------------------------------------------

    def plot_surface(self, res_j1_deg=60, res_j2=0.25, res_j3_deg=60, res_j4_deg=60, color='green'):
        """
        Sample configurations in joint space and scatter the reachable positions.
        Coarse but enough to visualise workspace.
        """
        points = []

        j1_min, j1_max = self.joint_limits[0]
        j2_min, j2_max = self.joint_limits[1]
        j3_min, j3_max = self.joint_limits[2]
        j4_min, j4_max = self.joint_limits[3]

        q1_range = np.radians(np.arange(j1_min, j1_max + 1e-3, res_j1_deg))
        q2_range = np.arange(j2_min, j2_max + 1e-6, res_j2)
        q3_range = np.radians(np.arange(j3_min, j3_max + 1e-3, res_j3_deg))
        q4_range = np.radians(np.arange(j4_min, j4_max + 1e-3, res_j4_deg))

        total_points = len(q1_range) * len(q2_range) * len(q3_range) * len(q4_range)
        current_point = 0

        for q1 in q1_range:
            for q2 in q2_range:
                for q3 in q3_range:
                    for q4 in q4_range:
                        q = [q1, q2, q3, q4]
                        self.forward_kinematics(q)
                        points.append(self.end_effector_position.copy())
                        current_point += 1
                        print(f"{current_point}/{total_points}", end="\r")

        points = np.array(points)
        scatter = self.ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            alpha=0.3,
            c=color,
            marker='.',
            s=1.0
        )
        self.workspace_plots.append(scatter)

    def plot_workspace(self):
        """
        Visualise the robot's workspace by sampling a coarse grid in joint space.
        """
        self.plot_surface()

    # ------------------------------------------------------------------
    # Block stacking animation
    # ------------------------------------------------------------------

    def animate_motion(self, q_start, q_end, steps=50):
        q_start = np.array(q_start)
        q_end = np.array(q_end)
        for s in np.linspace(0.0, 1.0, steps):
            q = (1 - s) * q_start + s * q_end
            self.set_joint_variables(q)
            self.env.step()

    def execute_block_stacking_sequence(self, steps_per_move=50):
        """
        Pick three blocks from the floor and place them on three shelves.
        Uses floor_targets[i] -> shelf_targets[i].
        """
        # Start at home
        self.set_joint_variables(self.q_home)

        for i in range(3):
            floor = self.floor_targets[i]
            shelf = self.shelf_targets[i]

            # Pre-grasp above floor block
            pre_floor = floor.copy()
            pre_floor[2] += 0.10  # 10 cm above

            q_curr = self.joint_variables.copy()

            q_pre_floor = self.inverse_kinematics(pre_floor)
            if q_pre_floor is None:
                print(f"Cannot reach pre-floor pose {i}")
                continue
            self.animate_motion(q_curr, q_pre_floor, steps_per_move)

            # Move down to grasp
            q_floor = self.inverse_kinematics(floor)
            if q_floor is None:
                print(f"Cannot reach floor pose {i}")
                continue
            self.animate_motion(q_pre_floor, q_floor, steps_per_move)
            # (Here you can imagine 'closing gripper')

            # Lift back up
            self.animate_motion(q_floor, q_pre_floor, steps_per_move)

            # Pre-place above shelf
            pre_shelf = shelf.copy()
            pre_shelf[2] += 0.10

            q_pre_shelf = self.inverse_kinematics(pre_shelf)
            if q_pre_shelf is None:
                print(f"Cannot reach pre-shelf pose {i}")
                continue
            self.animate_motion(q_pre_floor, q_pre_shelf, steps_per_move)

            # Place on shelf
            q_shelf = self.inverse_kinematics(shelf)
            if q_shelf is None:
                print(f"Cannot reach shelf pose {i}")
                continue
            self.animate_motion(q_pre_shelf, q_shelf, steps_per_move)
            # (Here you can imagine 'opening gripper')

            # Return to pre-shelf before next block
            self.animate_motion(q_shelf, q_pre_shelf, steps_per_move)

        # Back to home at the end
        self.animate_motion(self.joint_variables, self.q_home, steps_per_move)


if __name__ == '__main__':
    robot = RobotArm()
    robot.setup_visualization()
    robot.plot_robot_arm()

    print("Window running. Close the figure window or press Ctrl+C in the terminal to exit.")

    try:
        # Keep the script alive so the window doesn't close immediately
        while True:
            plt.pause(0.1)
    except KeyboardInterrupt:
        plt.close('all')
        sys.exit(0)

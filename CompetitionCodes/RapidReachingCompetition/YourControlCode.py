import time

import mujoco
import numpy as np


class YourCtrl:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, target_points):
        self.m = m
        self.d = d
        self.target_points = target_points
        self.num_points = target_points.shape[1]

        # Tracking  which points we've visited (start all as  False)
        self.visited = [False] * self.num_points
        self.visit_threshold = 0.01
        self.stuck_timeout = 20000  # 20 seconds - very patient

        # Getting end effector body ID
        self.ee_body_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame"
        )

        # Control gains
        self.kp_task = 800.0  # Moderate gain
        self.kd_task = 60.0  # Moderate damping

        # Control gains for null-space (DISABLED)
        self.kp_null = 0.0
        self.kd_null = 0.0

        # Desired null-space configuration
        self.q_null = np.array([0.0, -0.5, 0.0, -1.2, 0.0, 0.0])

        # Per-joint torque limits matching XML forcerange
        # [base_yaw, shoulder_pitch, shoulder_roll, elbow, wrist_pitch, wrist_roll]
        self.torque_limits = np.array([3.0, 15.0, 5.0, 10.0, 3.0, 0.1])

        # Track current target for debugging
        self.current_target = None
        self.stuck_counter = 0

        # Timing
        self.start_time = time.time()
        self.completion_time = None

    def reset(self):
        """Resets controller state for a new run"""
        self.visited = [False] * self.num_points
        self.current_target = None
        self.stuck_counter = 0
        self.start_time = time.time()
        self.completion_time = None
        print("[Controller] Reset - ready for new run!")

    def _get_ee_position(self):
        """Getting current end effector position in world frame"""
        return self.d.xpos[self.ee_body_id].copy()

    def _get_ee_velocity(self):
        """Getting current end effector velocity in world frame"""
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.ee_body_id)
        # Linear velocity = Jacobian * joint velocities
        return jacp @ self.d.qvel[:6]

    def _get_jacobian(self):
        """Getting the 3D position Jacobian for the end effector"""
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.ee_body_id)
        return jacp[:, :6]  # Only first 6 joints

    def _select_nearest_target(self):
        """
        Select the nearest unvisited target point.
        Marks points as visited when we get within 35mm.
        Auto-skips points if stuck for too long.
        """
        ee_pos = self._get_ee_position()

        # First pass: Marking any unvisited points we're close enough to as visited
        for i in range(self.num_points):
            if not self.visited[i]:
                dist = np.linalg.norm(ee_pos - self.target_points[:, i])
                if dist < self.visit_threshold:
                    self.visited[i] = True
                    self.stuck_counter = 0  # Reset stuck counter

                    # Changing color to green
                    site_id = self.m.site(f"point{i}").id
                    self.m.site_rgba[site_id] = np.array([0.0, 1.0, 0.0, 1.0])  # Green

                    print(
                        f"[Controller] ✓ Visited Point {i}! (distance: {dist*1000:.1f}mm)"
                    )
                    print(
                        f"[Controller] Progress: {sum(self.visited)}/{self.num_points} points visited"
                    )

        # Check if all visited
        if all(self.visited):
            if self.completion_time is None:
                self.completion_time = time.time() - self.start_time
                print("[Controller] 🎉 All points visited!")
                print(f"[Controller] ⏱️  Total time: {self.completion_time:.2f} seconds")
            return 0, 0.0

        # Second pass: Finding nearest unvisited point
        min_dist = float("inf")
        nearest_idx = 0

        for i in range(self.num_points):
            if not self.visited[i]:
                dist = np.linalg.norm(ee_pos - self.target_points[:, i])
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

        # Checking if we're stuck on the same target
        if self.current_target == nearest_idx:
            self.stuck_counter += 1

            if self.stuck_counter % 2000 == 0:
                print(
                    f"[Controller] ... working on Point {nearest_idx} (dist: {min_dist*1000:.1f}mm, time: {self.stuck_counter/1000:.0f}s)"
                )

            # If stuck for too long, skip this point
            if self.stuck_counter > self.stuck_timeout:
                print(
                    f"[Controller] ⚠️ Stuck on Point {nearest_idx} for {self.stuck_timeout/1000:.0f}s (dist: {min_dist*1000:.1f}mm) - marking as skipped"
                )
                self.visited[nearest_idx] = True

                # Changing color to yellow to show it was skipped
                site_id = self.m.site(f"point{nearest_idx}").id
                self.m.site_rgba[site_id] = np.array(
                    [1.0, 1.0, 0.0, 1.0]
                )  # Yellow (skipped)

                self.stuck_counter = 0
                # Recursively finding next target
                return self._select_nearest_target()
        else:
            # New target - resetting counter and print
            if self.current_target != nearest_idx:
                print(
                    f"[Controller] → Targeting Point {nearest_idx} (distance: {min_dist*1000:.1f}mm)"
                )
            self.current_target = nearest_idx
            self.stuck_counter = 0

        return nearest_idx, min_dist

    def CtrlUpdate(self):
        """Control update - it computes joint torques to reach targets"""

        # Selecting nearest unvisited target
        target_idx, dist_to_target = self._select_nearest_target()

        # If all points visited, just hold position with gravity compensation
        if dist_to_target == 0.0:
            # Compute gravity compensation to prevent falling
            qfrc_bias = np.zeros(self.m.nv)
            mujoco.mj_rne(self.m, self.d, 0, qfrc_bias)

            # Add damping to stop motion
            damping_torque = -20.0 * self.d.qvel[:6]

            return qfrc_bias[:6] + damping_torque

        target_pos = self.target_points[:, target_idx]

        # Get current end effector state
        ee_pos = self._get_ee_position()
        ee_vel = self._get_ee_velocity()

        # Task-space control: PD control in Cartesian space
        pos_error = target_pos - ee_pos

        # Desired task-space velocity (PD control)
        desired_task_vel = self.kp_task * pos_error - self.kd_task * ee_vel

        # Limit maximum task velocity
        max_task_vel = 1.5  # In  m/s - moderate speed
        task_vel_norm = np.linalg.norm(desired_task_vel)
        if task_vel_norm > max_task_vel:
            desired_task_vel = desired_task_vel * (max_task_vel / task_vel_norm)

        # Getting the  Jacobian
        J = self._get_jacobian()

        # Computing pseudo-inverse with damping
        damping = 0.02  # Moderate damping
        J_damped = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))

        # Desired joint velocities to achieve task velocity
        q_dot_desired = J_damped @ desired_task_vel

        # Null-space control: drive joints toward comfortable configuration
        # Note: This doesn't affect end effector motion but improves arm posture
        null_space_proj = np.eye(6) - J_damped @ J
        q_error_null = self.q_null - self.d.qpos[:6]
        q_dot_null = self.kp_null * q_error_null - self.kd_null * self.d.qvel[:6]

        # Combined desired joint velocity
        q_dot_cmd = q_dot_desired + null_space_proj @ q_dot_null

        kp_joint = 30.0  # Position gain at joint level
        kd_joint = 25.0  # Velocity damping at joint level

        # Computing desired joint positions using simple integration

        dt = 0.001  # MuJoCo timestep
        q_desired_integrated = self.d.qpos[:6] + q_dot_cmd * dt

        # PD control: position error + velocity error
        torque = kp_joint * (q_desired_integrated - self.d.qpos[:6]) + kd_joint * (
            q_dot_cmd - self.d.qvel[:6]
        )

        # Adding  gravity compensation
        qfrc_bias = np.zeros(self.m.nv)
        mujoco.mj_rne(self.m, self.d, 0, qfrc_bias)  # Computes gravity and Coriolis
        torque += qfrc_bias[:6]

        # Per-joint torque the limits matching XML forcerange
        for i in range(6):
            torque[i] = np.clip(
                torque[i], -self.torque_limits[i], self.torque_limits[i]
            )

        return torque

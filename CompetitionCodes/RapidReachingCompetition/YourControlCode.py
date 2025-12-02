import time

import mujoco
import numpy as np


class YourCtrl:

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, target_points):

        self.m = m

        self.d = d

        self.target_points = target_points

        self.num_points = target_points.shape[1]

        # Tracking which points we've visited (start all as False)

        self.visited = [False] * self.num_points

        self.visit_threshold = 0.025

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

        # NEW: per-point cooldown so we can temporarily avoid points

        self.avoid_counts = np.zeros(self.num_points, dtype=int)

        # Timing

        self.start_time = time.time()

        self.completion_time = None

    def reset(self):
        """Resets controller state for a new run"""

        self.visited = [False] * self.num_points

        self.current_target = None

        self.stuck_counter = 0

        self.avoid_counts = np.zeros(self.num_points, dtype=int)

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

        Marks points as visited when we get within visit_threshold.

        Auto-skips points if stuck for too long.

        """

        ee_pos = self._get_ee_position()

        # NEW: decrement cooldowns each call

        if np.any(self.avoid_counts > 0):

            self.avoid_counts = np.maximum(0, self.avoid_counts - 1)

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

                print("[Controller]  All points visited!")

                print(f"[Controller]  Total time: {self.completion_time:.2f} seconds")

            return 0, 0.0

        # Second pass: Finding nearest unvisited, non-cooled-down point

        min_dist = float("inf")

        nearest_idx = None

        for i in range(self.num_points):

            if (not self.visited[i]) and (self.avoid_counts[i] == 0):

                dist = np.linalg.norm(ee_pos - self.target_points[:, i])

                if dist < min_dist:

                    min_dist = dist

                    nearest_idx = i

        # If no candidate found (all unvisited are on cooldown), fall back to any unvisited

        if nearest_idx is None:

            for i in range(self.num_points):

                if not self.visited[i]:

                    dist = np.linalg.norm(ee_pos - self.target_points[:, i])

                    if dist < min_dist:

                        min_dist = dist

                        nearest_idx = i

        # Safety: if still None (shouldn't happen), just stop

        if nearest_idx is None:

            return 0, 0.0

        # Checking if we're stuck on the same target

        if self.current_target == nearest_idx:

            self.stuck_counter += 1

            if self.stuck_counter % 2000 == 0:

                print(
                    f"[Controller] ... working on Point {nearest_idx} "
                    f"(dist: {min_dist*1000:.1f}mm, time: {self.stuck_counter/1000:.0f}s)"
                )

            # NEW: if stuck for ~0.5s, temporarily avoid this point and switch target

            if self.stuck_counter > 500:

                # Find alternative unvisited, non-cooled-down point

                alt_idx = None

                alt_dist = float("inf")

                for j in range(self.num_points):

                    if (
                        (not self.visited[j])
                        and (j != nearest_idx)
                        and (self.avoid_counts[j] == 0)
                    ):

                        d = np.linalg.norm(ee_pos - self.target_points[:, j])

                        if d < alt_dist:

                            alt_dist = d

                            alt_idx = j

                if alt_idx is not None:

                    # Put current point on cooldown so we can revisit later with new posture

                    self.avoid_counts[nearest_idx] = 1500  # ~1.5 seconds at 1kHz

                    print(
                        f"[Controller] Stuck on Point {nearest_idx} "
                        f"(dist: {min_dist*1000:.1f}mm) - switching to Point {alt_idx} and will revisit later"
                    )

                    self.current_target = alt_idx

                    self.stuck_counter = 0

                    return alt_idx, alt_dist

                # If no alternative exists, fall through to timeout logic below

            # If stuck for too long, skip this point (original behavior)

            if self.stuck_counter > self.stuck_timeout:

                print(
                    f"[Controller] Stuck on Point {nearest_idx} for {self.stuck_timeout/1000:.0f}s "
                    f"(dist: {min_dist*1000:.1f}mm) - marking as skipped"
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

        max_task_vel = 2.5  # In m/s - moderate speed

        task_vel_norm = np.linalg.norm(desired_task_vel)

        if task_vel_norm > max_task_vel:

            desired_task_vel = desired_task_vel * (max_task_vel / task_vel_norm)

        # Getting the Jacobian

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

        dt = self.m.opt.timestep  # use model timestep

        q_desired_integrated = self.d.qpos[:6] + q_dot_cmd * dt

        # PD control: position error + velocity error

        torque = kp_joint * (q_desired_integrated - self.d.qpos[:6]) + kd_joint * (
            q_dot_cmd - self.d.qvel[:6]
        )

        # Adding gravity compensation

        qfrc_bias = np.zeros(self.m.nv)

        mujoco.mj_rne(self.m, self.d, 0, qfrc_bias)  # Computes gravity and Coriolis

        torque += qfrc_bias[:6]

        # Per-joint torque limits matching XML forcerange

        for i in range(6):

            torque[i] = np.clip(
                torque[i], -self.torque_limits[i], self.torque_limits[i]
            )

        return torque

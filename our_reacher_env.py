from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv
from our_reacher import OurReacher
import numpy as np

class OurReacherEnv(ReacherBulletEnv):
    def __init__(self):
        self.robot = OurReacher()
        BaseBulletEnv.__init__(self, self.robot)

    def our_step(self, a, g):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.our_calc_state(g)  # sets self.to_target_vec

        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        epsilon = [0.1, 0.1]
        within_goal = self.robot.to_target_vec < epsilon

        electricity_cost = (
                -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot))  # work torque*angular_velocity
                - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
        self.rewards = [100 if within_goal else 0, float(electricity_cost), float(stuck_joint_cost)]
        self.HUD(state, a, False)
        return state, sum(self.rewards), False, {}
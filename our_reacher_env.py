from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv
from our_reacher import OurReacher
import numpy as np

class OurReacherEnv(ReacherBulletEnv):
    def __init__(self):
        self.robot = OurReacher()
        BaseBulletEnv.__init__(self, self.robot)
        self.epsilon = 1e-3

    # g should be shape (9,)
    def our_step(self, a, g):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.our_calc_state(g)  # sets self.to_target_vec

        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        within_goal = all(np.abs(self.robot.to_target_vec) < self.epsilon)

        electricity_cost = (
                -0.10 * (np.abs(a[0] * self.robot.theta_dot) + np.abs(a[1] * self.robot.gamma_dot))  # work torque*angular_velocity
                - 0.01 * (np.abs(a[0]) + np.abs(a[1]))  # stall torque require some energy
        )
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0
        # self.rewards = [100 if within_goal else 0, float(electricity_cost), float(stuck_joint_cost)]
        self.rewards = [100 if within_goal else 0]
        self.HUD(state, a, False)
        return state, sum(self.rewards), all(np.abs(self.robot.to_target_vec) < 1e-4), {}

    # binary goal-conditioned reward function
    # implicitly, state = state' after taking some action
    def goal_cond_reward(self, state, goal):
        assert goal.shape[0] == 2

        to_goal_vec = np.array(state[0] + state[2], state[1] + state[3]) - goal
        within_goal = all(np.abs(to_goal_vec) < self.epsilon)

        # maybe make this 1???
        return 100 if within_goal else 0

    def sample_goal_state(self, sigma=1e-3):
        return np.random.normal(self.robot.target.pose().xyz()[:2], sigma)
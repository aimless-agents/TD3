from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv
from our_reacher import OurReacher
import numpy as np

class OurReacherEnv(ReacherBulletEnv):
    def __init__(self, epsilon=4e-2, render=False):
        self.robot = OurReacher()
        BaseBulletEnv.__init__(self, self.robot, render)

        self.epsilon = epsilon
        self.g = np.zeros(2)
        self.original_rewards = None

    def set_goal(self, g):
        self.g = g

    def set_epsilon(self, eps):
        self.epsilon = eps

    def step(self, a):
        assert (not self.scene.multiplayer)
        self.robot.apply_action(a)
        self.scene.global_step()

        next_state = self.robot.our_calc_state(self.g)  # sets self.to_target_vec

        rewards = self.goal_cond_reward(a, next_state, self.g)
        
        self.HUD(next_state, a, False)
        return next_state, rewards, False, {}

    # binary goal-conditioned reward function
    # models reward from the transition state' (implicit) -> state
    def goal_cond_reward(self, prev_action, next_state, goal):
        assert goal.shape[0] == 2

        to_goal_vec = np.array(next_state[0] + next_state[2], next_state[1] + next_state[3]) - goal
        within_goal = np.sum(to_goal_vec ** 2) < self.epsilon

        electricity_cost = (
                -0.10 * (np.abs(prev_action[0] * next_state[-3]) + np.abs(prev_action[1] * next_state[-1]))  # work torque*angular_velocity
                - 0.01 * (np.abs(prev_action[0]) + np.abs(prev_action[1]))  # stall torque require some energy
        )
        electricity_exceeded = electricity_cost > 0.01
        stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        self.original_rewards = sum([float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)])
        # maybe make this 1???
        return 0 if within_goal and stuck_joint_cost == 0 and not electricity_exceeded else -5

    # pass in sigma=0 to get the original target goal state
    def sample_goal_state(self, sigma=1e-3):
        return np.random.normal(self.robot.target.pose().xyz()[:2], sigma)
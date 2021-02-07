from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv
from our_reacher import OurReacher
import numpy as np
from  gym.envs.mujoco import ReacherEnv

# built on top of https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher.py implementation
# xml is https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/reacher.xml
class OurReacherEnv(ReacherEnv):
    def __init__(self, epsilon=4e-2, render=False):
        # self.robot = OurReacher()

        self.epsilon = epsilon
        self.g = np.zeros(2)
        self.original_rewards = None
        super().__init__()

    def set_goal(self, g):
        self.g = g

    def set_epsilon(self, eps):
        self.epsilon = eps

    def step(self, a):
        # assert (not self.scene.multiplayer)
        # self.robot.apply_action(a)
        # self.scene.global_step()

        # next_state = self.robot.our_calc_state(self.g)  # sets self.to_target_vec

        # rewards = self.goal_cond_reward(a, next_state, self.g)
        
        # self.HUD(next_state, a, False)
        # return next_state, rewards, False, {}

        
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        reward = self.goal_cond_reward(a, ob, self.g)

        done = False
        return ob, reward, done, {}
    
    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip"), 
            np.concatenate([self.g, [0.01]])
        ])
    # binary goal-conditioned reward function
    # models reward from the transition state' (implicit) -> state
    def goal_cond_reward(self, prev_action, next_state, goal):
        to_goal_vec = self.get_fingertip_from_state(next_state) \
                        - self.get_goal_from_state(next_state) # current position of fingertip - goal position
        reward_dist = - np.linalg.norm(to_goal_vec)
        reward_ctrl = - np.square(prev_action).sum()
        self.original_rewards = reward_dist + reward_ctrl
        within_goal = 0 if np.abs(self.original_rewards) < self.epsilon else -1
        return within_goal
        
        # assert goal.shape[0] == 2

        # to_goal_vec = np.array(next_state[0] + next_state[2], next_state[1] + next_state[3]) - goal
        # within_goal = np.sum(to_goal_vec ** 2) < self.epsilon

        # electricity_cost = (
        #         -0.10 * (np.abs(prev_action[0] * next_state[-3]) + np.abs(prev_action[1] * next_state[-1]))  # work torque*angular_velocity
        #         - 0.01 * (np.abs(prev_action[0]) + np.abs(prev_action[1]))  # stall torque require some energy
        # )
        # electricity_exceeded = electricity_cost > 0.01
        # stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0

        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        # self.original_rewards = sum([float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)])
        # # maybe make this 1???
        # return 0.1 if within_goal and stuck_joint_cost == 0 and not electricity_exceeded else 0

    # pass in sigma=0 to get the original target goal state
    def sample_goal_state(self, sigma=1e-3):
        return np.random.normal(self.get_body_com("target")[:2], sigma)

    def within_goal(self, state, eps=1e-3):
        return np.linalg.norm(self.get_fingertip_from_state(state) - self.get_body_com("target")) < eps

    def get_fingertip_from_state(self, state):
        return state[-6:-3]
    def get_goal_from_state(self, state):
        return state[-3:]

# class OurReacherEnv(ReacherBulletEnv):
#     def __init__(self, epsilon=4e-2, render=False):
#         self.robot = OurReacher()
#         BaseBulletEnv.__init__(self, self.robot, render)

#         self.epsilon = epsilon
#         self.g = np.zeros(2)
#         self.original_rewards = None

#     def set_goal(self, g):
#         self.g = g

#     def set_epsilon(self, eps):
#         self.epsilon = eps

#     def step(self, a):
#         assert (not self.scene.multiplayer)
#         self.robot.apply_action(a)
#         self.scene.global_step()

#         next_state = self.robot.our_calc_state(self.g)  # sets self.to_target_vec

#         rewards = self.goal_cond_reward(a, next_state, self.g)
        
#         self.HUD(next_state, a, False)
#         return next_state, rewards, False, {}

#     # binary goal-conditioned reward function
#     # models reward from the transition state' (implicit) -> state
#     def goal_cond_reward(self, prev_action, next_state, goal):
#         assert goal.shape[0] == 2

#         to_goal_vec = np.array(next_state[0] + next_state[2], next_state[1] + next_state[3]) - goal
#         within_goal = np.sum(to_goal_vec ** 2) < self.epsilon

#         electricity_cost = (
#                 -0.10 * (np.abs(prev_action[0] * next_state[-3]) + np.abs(prev_action[1] * next_state[-1]))  # work torque*angular_velocity
#                 - 0.01 * (np.abs(prev_action[0]) + np.abs(prev_action[1]))  # stall torque require some energy
#         )
#         electricity_exceeded = electricity_cost > 0.01
#         stuck_joint_cost = -0.1 if np.abs(np.abs(self.robot.gamma) - 1) < 0.01 else 0.0

#         potential_old = self.potential
#         self.potential = self.robot.calc_potential()
#         self.original_rewards = sum([float(self.potential - potential_old), float(electricity_cost), float(stuck_joint_cost)])
#         # maybe make this 1???
#         return 0.1 if within_goal and stuck_joint_cost == 0 and not electricity_exceeded else 0

#     # pass in sigma=0 to get the original target goal state
#     def sample_goal_state(self, sigma=1e-3):
#         return np.random.normal(self.robot.target.pose().xyz()[:2], sigma)

#     def within_goal(self, state, eps=1e-3):
#         return np.sum(([state[0] + state[2], state[1] + state[3]] - self.robot.target.pose().xyz()[:2]) ** 2) < eps
from pybulletgym.envs.roboschool.envs.env_bases import BaseBulletEnv
from pybulletgym.envs.roboschool.envs.manipulation.reacher_env import ReacherBulletEnv
from our_reacher import OurReacher
import numpy as np

class OurReacherEnv(ReacherBulletEnv):
    def __init__(self):
        self.robot = OurReacher()
        BaseBulletEnv.__init__(self, self.robot)
        self.epsilon = 1e-3
        self._max_episode_steps = 150    # copied manually from ReacherEnv

    # g should be shape (9,)
    def our_step(self, a, g):
        assert (not self.scene.multiplayer)
        assert g.shape[0] == 9
        self.robot.apply_action(a)
        self.scene.global_step()

        state = self.robot.our_calc_state(g)  # sets self.to_target_vec
        within_goal = all(np.abs(self.robot.to_target_vec) < self.epsilon)

        self.rewards = [100 if within_goal else 0]
        self.HUD(state, a, False)
        return state, sum(self.rewards), all(np.abs(self.robot.to_target_vec) < 1e-4), {}

    # binary goal-conditioned reward function
    # models reward from the transition state' (implicit) -> state
    def goal_cond_reward(self, state, goal):
        assert goal.shape[0] == 2

        to_goal_vec = np.array(state[0] + state[2], state[1] + state[3]) - goal
        within_goal = all(np.abs(to_goal_vec) < self.epsilon)

        # maybe make this 1???
        return 100 if within_goal else 0

    # pass in sigma=0 to get the original target goal state
    def sample_goal_state(self, sigma=1e-3):
        return np.random.normal(self.robot.target.pose().xyz()[:2], sigma)
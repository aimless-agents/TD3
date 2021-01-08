from pybulletgym.envs.roboschool.robots.manipulators.reacher import Reacher
import numpy as np

class OurReacher(Reacher):
    def __init__(self):
        super().__init__()

    def our_calc_state(self, g):
        import pdb; pdb.set_trace()
        theta, self.theta_dot = self.central_joint.current_relative_position()
        self.gamma, self.gamma_dot = self.elbow_joint.current_relative_position()
        target_x, _ = self.jdict["target_x"].current_position()
        target_y, _ = self.jdict["target_y"].current_position()
        self.to_target_vec = np.array(self.fingertip.pose().xyz()) - np.array([g[0], g[1], 0.01])
        return np.array([
            target_x,
            target_y,
            self.to_target_vec[0],
            self.to_target_vec[1],
            np.cos(theta),
            np.sin(theta),
            self.theta_dot,
            self.gamma,
            self.gamma_dot,
        ])

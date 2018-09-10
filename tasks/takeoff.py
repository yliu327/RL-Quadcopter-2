import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=[0.,0.,0.,0.,0.,0.], init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pose = None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal0
        self.target_pos = target_pose[:3] if target_pose is not None else np.array([0,0,10])
        self.target_vel = target_pose[3:] if target_pose is not None else np.array([0,0,0])
        self.best_pose = init_pose
        self.best_reward = -np.inf
        
        self.vel_w = 0
        self.pos_w = 1

    def get_reward(self):
        """Uses current pose of sim to return reward."""
       
        pos_error = np.sum(abs(self.sim.pose[:3] - self.target_pos[:3]))
        pos_error = np.log(pos_error)
        z_error = abs(self.sim.pose[2] - self.target_pos[2])
        velocity_error = np.dot(np.subtract(1, np.tanh(self.sim.pose[:3])), self.sim.v)
        reward = 1. - pos_error - 0.02 * z_error
        #reward = 1 - z_error - xy_erro, r/800 - ((1-z_error)*z_v/100) - angv/20
        reward = np.clip(reward, -2, None)

        #reward = np.maximum(np.minimum(reward, max_reward), min_reward)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            if done:
                if reward > self.best_reward:
                    self.best_pose = self.sim.pose
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
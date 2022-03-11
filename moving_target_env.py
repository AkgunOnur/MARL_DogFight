import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from gym.envs.classic_control import rendering
import math
import time



class MovingTarget(gym.Env):
    def __init__(self, dog_fight_range=25.0, detection_angle=60.0, opponent_range=25.0, opponent_angle=60.0, name="A", opponent="B", locked_reward=1.0, get_locked_reward=-0.8, max_timesteps=2000, visualization = False, level = None):
        self.dt = .05
        self.max_angular_velocity = 1.0
        self.min_velocity = 1.0
        self.max_velocity = 8.0
        self.x_target = 0.
        self.y_target = 0.
        self.v_target = 0.0
        self.psi_target = 0.
        self.psi_goal = 0.
        self.viewer = None
        self.dog_fight_range = dog_fight_range
        self.detection_angle = detection_angle
        self.opponent_range = opponent_range
        self.opponent_angle = opponent_angle
        self.name = name
        self.opponent = opponent
        self.state_opponent = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.map_lim = 50.0
        self.locked_reward = locked_reward
        self.get_locked_reward = get_locked_reward
        self.state1 = None
        self.state2 = None
        self.name1 = name
        self.name2 = opponent
        self.reward_range = 2
        self.visualization = visualization
        self.level = level
        self.max_timesteps = max_timesteps
        self.timesteps = 0
        # self.metadata = {'render.modes': ['human']}

        # self.action_space = spaces.Box(
        #     low=np.array([-np.float32(self.max_angular_velocity), np.float32(self.min_velocity)]),
        #     high=np.array([np.float32(self.max_angular_velocity), np.float32(self.max_velocity)]), shape=(2,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Box(
            np.array([-self.max_angular_velocity, self.min_velocity]).astype(np.float32),
            np.array([self.max_angular_velocity, self.max_velocity]).astype(np.float32))

        high = np.array([-1] * 11).astype(np.float32)

        self.observation_space = spaces.Box(low=-high, high=high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_target(self, x_target, y_target, v_target=1.0):
        self.x_target = x_target
        self.y_target = y_target
        self.v_target = v_target
        self.psi_target = self.np_random.uniform(low=-np.pi, high=np.pi)


    def dubin_model(self, x1, y1, psi1, u_angular, u_linear):
        #x_dot = v*cos(psi)
        #y_dot = v*sin(psi)
        #psi_dot = u

        psi1_dot = u_angular
        psi1 = psi1 + psi1_dot * self.dt
        psi1 = angle_normalize(psi1)
        x1_dot = u_linear*np.cos(psi1 + np.pi/2)
        x1 = x1 + x1_dot * self.dt
        y1_dot = u_linear*np.sin(psi1 + np.pi/2)
        y1 = y1 + y1_dot * self.dt

        return x1, y1, psi1, x1_dot, y1_dot


    def step(self, action1):
        self.timesteps += 1
        dt = self.dt
        info = dict()
        done = False
        locked, get_locked = False, False
        reward = 0

        x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, psi1_dot, distance, diff_angle, diff_angle_1 = self.state1 
        #Target moves
        self.x_target += (self.v_target * np.cos(self.psi_target + np.pi/2) * dt)
        self.y_target += (self.v_target * np.sin(self.psi_target + np.pi/2) * dt)

        if abs(self.x_target) >=  self.map_lim or abs(self.y_target) >=  self.map_lim:
            self.psi_target += (np.pi*10/12) 

        u_angular1, u_linear1 = action1

        x1, y1, psi1, x1_dot, y1_dot = self.dubin_model(x1, y1, psi1, u_angular1, u_linear1)

        x1_diff = self.x_target - x1
        y1_diff = self.y_target - y1
        psi1_diff = self.psi_target - psi1
    
        distance = np.sqrt(x1_diff**2 + y1_diff**2)
        max_dist = np.sqrt(2)*2*self.map_lim
        
        start_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 - self.detection_angle/2*np.pi/180)
        end_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 + self.detection_angle/2*np.pi/180)
        diff_angle_1 = angle_normalize_2pi(np.arctan2(self.y_target-y1, self.x_target-x1))

        start_angle_2 = angle_normalize_2pi(self.psi_target + np.pi/2 - self.opponent_angle/2*np.pi/180)
        end_angle_2 = angle_normalize_2pi(self.psi_target + np.pi/2 + self.opponent_angle/2*np.pi/180)
        diff_angle_2 = angle_normalize_2pi(np.arctan2(y1-self.y_target, x1-self.x_target))

        diff_angle = angle_normalize(psi1-self.psi_target)

        if distance >= 50:
            reward = -1.0
            done = True
        elif x1 > 1.25*self.map_lim or x1 < -1.25*self.map_lim or y1 > 1.25*self.map_lim or y1 < -1.25*self.map_lim:
            reward = -1.0
            done = True
        elif x1 > self.map_lim or x1 < -self.map_lim or y1 > self.map_lim or y1 < -self.map_lim:
            reward = -1.0
            # done = True
            # print (f"\nPlane {self.name1} is out of map!")
        elif distance <= 5.0:
            reward = -0.8
            # print (f"\nPlane {self.name1} and {self.name2} crashed!") 
        else:
            # reward = np.clip(-0.4*distance / max_dist -0.6*np.abs(diff_angle_1)/(2*np.pi), -1.0, 1.0)
            reward = np.clip(-0.5*distance / max_dist - 0.5*np.abs(diff_angle) / np.pi + 0.2 * u_linear1/self.max_velocity, -1.0, 1.0)
            if distance <= self.dog_fight_range:
                if end_angle_1 > start_angle_1:
                    if start_angle_1 < diff_angle_1 < end_angle_1:
                        #print ("The opponent " + self.opponent + " is seen!")
                        locked = True
                        reward = self.locked_reward
                else:
                    if diff_angle_1 > start_angle_1 or diff_angle_1 < end_angle_1:
                        #print ("The opponent " + self.opponent + " is seen!")
                        locked = True
                        reward = self.locked_reward

            if distance <= self.opponent_range:
                if end_angle_2 > start_angle_2:
                    if start_angle_2 < diff_angle_2 < end_angle_2:
                        #print ("The plane " + self.name + " is seen!")
                        get_locked = True
                        reward = self.get_locked_reward
                else:
                    if diff_angle_2 > start_angle_2 or diff_angle_2 < end_angle_2:
                        #print ("The plane " + self.name + " is seen!")
                        get_locked = True
                        reward = self.get_locked_reward

            if locked and get_locked:
                reward = 0.0
        
        
        if self.timesteps >= self.max_timesteps:
            done = True


        # To restrain the position of plane
        # x1 = np.clip(x1, -self.map_lim, self.map_lim)
        # y1 = np.clip(y1, -self.map_lim, self.map_lim)

        if self.visualization:
            self.render()  
            # time.sleep(0.01)
            if done:
                self.close()

        self.state1 = (x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, u_angular1, distance, diff_angle, diff_angle_1)

        # print (f"reward: {reward:.4} diff: {diff_angle*180/np.pi:.4} angular: {action1[0]:.4} linear: {action1[1]:.4}")
        # time.sleep(0.1)
        return self._get_obs(self.state1), reward, done, info


                

    def _get_obs(self, state1):
        pos_coeff = self.map_lim
        vel_coeff = self.max_velocity
        angle_coeff = np.pi
            
        x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, u_angular, distance, diff_angle, diff_angle_1 = state1

        obs_state1 = np.array([x1/pos_coeff, y1/pos_coeff, x1_diff/pos_coeff, y1_diff/pos_coeff, psi1_diff/angle_coeff, 
                                x1_dot/vel_coeff, y1_dot/vel_coeff, u_angular, distance/(np.sqrt(2)*2*pos_coeff), 
                                diff_angle/angle_coeff, diff_angle_1/(2*angle_coeff)])

        # print ("obs: ", obs_state1)
        return obs_state1

    def reset(self):
        self.timesteps = 0
        x_ind = np.random.randint(0,5)
        y_ind = np.random.randint(0,5)

        map_lim_1 = -self.map_lim + x_ind * 20
        map_lim_2 = -self.map_lim + (x_ind + 1) * 20
        map_lim_1 = -self.map_lim + y_ind * 20
        map_lim_2 = -self.map_lim + (y_ind + 1) * 20

        # x_ind = np.random.randint(0,5)
        # y_ind = np.random.randint(0,5)

        x1 = self.np_random.uniform(low=map_lim_1, high=map_lim_2)
        y1 = self.np_random.uniform(low=map_lim_1, high=map_lim_2)
        psi1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        x1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        y1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        psi1_dot = self.np_random.uniform(low=-0.5, high=0.5)

        self.x_target = self.np_random.uniform(low=map_lim_1, high=map_lim_2)
        self.y_target = self.np_random.uniform(low=map_lim_1, high=map_lim_2)
        self.psi_target = self.np_random.uniform(low=-np.pi, high=np.pi)

        if self.level == "level1":
            self.v_target = 0
        else:
            self.v_target = self.np_random.uniform(low=1.0, high=6.0)
        diff_angle = angle_normalize(psi1-self.psi_target)

        distance = np.sqrt((self.x_target - x1)**2 + (self.y_target - y1)**2)

        self.state1 = np.array([x1, y1, psi1, self.x_target - x1, self.y_target - y1, self.psi_target - psi1, x1_dot, y1_dot, psi1_dot, distance, diff_angle, 0.])

        return self._get_obs(state1 = self.state1)


    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.map_lim, self.map_lim, -self.map_lim, self.map_lim)

            target = rendering.make_circle(2)
            target.set_color(.1, .1, .8)
            self.plane1_transform = rendering.Transform()
            self.plane2_transform = rendering.Transform()
            self.target_transform = rendering.Transform()
            target.add_attr(self.target_transform)
            # if self.scenario != "dog_fight":
            #     self.viewer.add_geom(target)
            
            fname1 = path.join(path.dirname(__file__), "assets/plane_A.png")
            fname2 = path.join(path.dirname(__file__), "assets/plane_B.png")
            self.plane1 = rendering.Image(fname1, 7., 7.)
            self.plane2 = rendering.Image(fname2, 7., 7.)
            #self.imgtrans = rendering.Transform()
            self.plane1.add_attr(self.plane1_transform)
            self.plane2.add_attr(self.plane2_transform)

        self.viewer.add_onetime(self.plane1)
        self.plane1_transform.set_translation(self.state1[0], self.state1[1])
        self.plane1_transform.set_rotation(self.state1[2])

        
        # self.target_transform.set_rotation(self.psi_target)
        # self.target_transform.set_translation(self.x_target, self.y_target)
        self.viewer.add_onetime(self.plane2)
        self.plane2_transform.set_translation(self.x_target, self.y_target)
        self.plane2_transform.set_rotation(self.psi_target)

    
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def angle_normalize_2pi(x):
    return (x % (2*np.pi))
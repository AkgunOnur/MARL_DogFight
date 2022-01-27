import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from gym.envs.classic_control import rendering
import math



class Dubin:
    def __init__(self, scenario="moving_target", dog_fight_range=25.0, detection_angle=60.0, opponent_range=25.0, opponent_angle=60.0, name="A", opponent="B", locked_reward=50.0, get_locked_reward=-40.0):
        self.dt = .05
        self.max_angular_velocity = 1.0
        self.min_velocity = 1.5
        self.max_velocity = 6.0
        self.x_target = 0.
        self.y_target = 0.
        self.v_target = 2.0
        self.psi_target = 0.
        self.psi_goal = 0.
        self.viewer = None
        self.scenario = scenario
        self.dog_fight_range = dog_fight_range
        self.detection_angle = detection_angle
        self.opponent_range = opponent_range
        self.opponent_angle = opponent_angle
        self.name = name
        self.opponent = opponent
        self.state_opponent = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.x_lim = 50.0
        self.y_lim = 50.0
        self.locked_reward = locked_reward
        self.get_locked_reward = get_locked_reward
        self.state1 = None
        self.state2 = None
        self.name1 = name
        self.name2 = opponent

        high = np.array([self.x_lim, self.y_lim, np.pi], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.max_angular_velocity, self.min_velocity]),
            high=np.array([self.max_angular_velocity, self.max_velocity]), shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

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


    def step(self, action1, action2=None):
        dt = self.dt
        # u_angular = np.clip(u_angular, -self.max_angular_velocity, self.max_angular_velocity)
        # u_linear = np.clip(u_linear, 1., self.max_velocity)
        done = False
        locked, get_locked = False, False
        reward = 0

        done1, done2 = False, False
        locked1, locked2 = False, False
        reward1, reward2 = 0, 0
        if self.scenario == "moving_target":
            x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, psi1_dot, distance, diff_angle_1, diff_angle_2 = self.state1 
            #Target moves
            self.x_target += (self.v_target * np.cos(self.psi_target + np.pi/2) * dt)
            self.y_target += (self.v_target * np.sin(self.psi_target + np.pi/2) * dt)

            if abs(self.x_target) >=  self.observation_space.high[0] or abs(self.y_target) >=  self.observation_space.high[1]:
                self.psi_target += (np.pi*10/12) 

            x_target = self.x_target
            y_target = self.y_target
            psi_target = self.psi_target

            u_angular1, u_linear1 = action1
            x1, y1, psi1, x1_dot, y1_dot = self.dubin_model(x1, y1, psi1, u_angular1, u_linear1)

            x1_diff = x_target - x1
            y1_diff = y_target - y1
            psi1_diff = psi_target - psi1
        
            distance = np.sqrt(x1_diff**2 + y1_diff**2)
            
            start_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 - self.detection_angle/2*np.pi/180)
            end_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 + self.detection_angle/2*np.pi/180)
            diff_angle_1 = angle_normalize_2pi(np.arctan2(y_target-y1, x_target-x1))

            start_angle_2 = angle_normalize_2pi(psi_target + np.pi/2 - self.opponent_angle/2*np.pi/180)
            end_angle_2 = angle_normalize_2pi(psi_target + np.pi/2 + self.opponent_angle/2*np.pi/180)
            diff_angle_2 = angle_normalize_2pi(np.arctan2(y1-y_target, x1-x_target))

            if x1 > self.x_lim or x1 < -self.x_lim or y1 > self.y_lim or y1 < -self.y_lim:
                reward = -1000.0
                done = True
                print (f"\n Plane {self.name1} is out of map!")
            elif distance <= 3.0:
                reward = -500.0
                # done = True
                # print (f"\n Plane {self.name1} and {self.name2} crashed!")
            else:
                reward = -3*distance - 2*abs(u_angular1) - abs(u_linear1)
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
            
            

            # To restrain the position of plane
            # x1 = np.clip(x1, self.observation_space.low[0], self.observation_space.high[0])
            # y1 = np.clip(y1, self.observation_space.low[1], self.observation_space.high[1])

            self.state1 = (x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, u_angular1, distance, diff_angle_1, diff_angle_2)

            return self._get_obs(self.state1), reward, done, {}


        elif self.scenario == "dog_fight":
            x1, y1, psi1, diff_x1, diff_y1, diff_psi1, x1_dot, y1_dot, psi1_dot, distance, diff_angle_1, diff_angle_2 = self.state1 
            x2, y2, psi2, diff_x2, diff_y2, diff_psi2, x2_dot, y2_dot, psi2_dot, distance, diff_angle_1, diff_angle_2 = self.state2

            u_angular1, u_linear1 = action1
            u_angular2, u_linear2 = action2

            x1, y1, psi1, x1_dot, y1_dot = self.dubin_model(x1, y1, psi1, u_angular1, u_linear1)
            x2, y2, psi2, x2_dot, y2_dot = self.dubin_model(x2, y2, psi2, u_angular2, u_linear2)

            

            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            diff_x1, diff_y1, diff_psi1 = x1 - x2, y1 - y2, angle_normalize(psi1 - psi2)
            diff_x2, diff_y2, diff_psi2 = x2 - x1, y2 - y1, angle_normalize(psi2 - psi1)

            start_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 - self.detection_angle/2*np.pi/180)
            end_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 + self.detection_angle/2*np.pi/180)
            diff_angle_1 = angle_normalize_2pi(np.arctan2(y2-y1, x2-x1))

            start_angle_2 = angle_normalize_2pi(psi2 + np.pi/2 - self.opponent_angle/2*np.pi/180)
            end_angle_2 = angle_normalize_2pi(psi2 + np.pi/2 + self.opponent_angle/2*np.pi/180)
            diff_angle_2 = angle_normalize_2pi(np.arctan2(y1-y2, x1-x2))

            reward1 = -distance - 2.5*abs(u_angular1) - 0.5*abs(u_linear1)
            reward2 = -distance - 2.5*abs(u_angular2) - 0.5*abs(u_linear2)
            
            if x2 > self.x_lim or x2 < -self.x_lim or y2 > self.y_lim or y2 < -self.y_lim:
                reward2 = -150.0
                done1, done2 = False, False
                # print (f"\n Plane {self.name2} is out of map!")
                x2 = np.clip(x2, -self.x_lim, self.x_lim)
                y2 = np.clip(y2, -self.y_lim, self.y_lim)
                
            if x1 > self.x_lim or x1 < -self.x_lim or y1 > self.y_lim or y1 < -self.y_lim:
                reward1 = -150.0
                done1, done2 = False, False
                # print (f"\n Plane {self.name1} is out of map!")
                x1 = np.clip(x1, -self.x_lim, self.x_lim)
                y1 = np.clip(y1, -self.y_lim, self.y_lim)
            elif distance <= 3.0:
                reward1, reward2 = -50.0, -50.0
                done1, done2 = False, False
                # print (f"\n Plane {self.name1} and {self.name2} crashed!")
            else:
                if distance <= self.dog_fight_range:
                    if end_angle_1 > start_angle_1:
                        if start_angle_1 < diff_angle_1 < end_angle_1:
                            #print ("The opponent " + self.opponent + " is seen!")
                            locked1 = True
                            reward1 = self.locked_reward
                            reward2 = self.get_locked_reward
                    else:
                        if diff_angle_1 > start_angle_1 or diff_angle_1 < end_angle_1:
                            #print ("The opponent " + self.opponent + " is seen!")
                            locked1 = True
                            reward1 = self.locked_reward
                            reward2 = self.get_locked_reward

                if distance <= self.opponent_range:
                    if end_angle_2 > start_angle_2:
                        if start_angle_2 < diff_angle_2 < end_angle_2:
                            #print ("The plane " + self.name + " is seen!")
                            locked2 = True
                            reward2 = self.locked_reward
                            reward1 = self.get_locked_reward
                    else:
                        if diff_angle_2 > start_angle_2 or diff_angle_2 < end_angle_2:
                            #print ("The plane " + self.name + " is seen!")
                            locked2 = True
                            reward2 = self.locked_reward
                            reward1 = self.get_locked_reward

                if locked1 and locked2:
                    reward1, reward2 = 0.0, 0.0

            

            self.state1 = (x1, y1, psi1, diff_x1, diff_y1, diff_psi1, x1_dot, y1_dot, psi1_dot, distance, diff_angle_1, diff_angle_2)
            self.state2 = (x2, y2, psi2, diff_x2, diff_y2, diff_psi2, x2_dot, y2_dot, psi2_dot, distance, diff_angle_1, diff_angle_2)

            return self._get_obs(self.state1,self.state2), (reward1, reward2), (done1, done2), {}
        

        # if self.scenario == "dog_fight":
        #     # if x1 > self.observation_space.high[0] or x1 < -self.observation_space.high[0]:
        #     #     x1 = np.random.uniform(low=-self.observation_space.high[0]/2, high=self.observation_space.high[0]/2)
        #     # if y1 > self.observation_space.high[1] or y1 < -self.observation_space.high[1]:
        #     #     y1 = np.random.uniform(low=-self.observation_space.high[1]/2, high=self.observation_space.high[1]/2)

        #     self.state = np.array([x1, y1, psi1, diff_x1, diff_y1, diff_psi1, x1_dot, y1_dot, psi1_dot, distance, diff_angle_1, diff_angle_2])
        # else:
        #     # self.state = np.array([x1, y1, psi1, x1_dot, y1_dot, psi1_dot, x1_diff, y1_diff, psi1_diff])
        #     self.state = np.array([x1, y1, psi1, diff_x1, diff_y1, diff_psi1, x1_dot, y1_dot, psi1_dot, distance, diff_angle_1, diff_angle_2])

        

    def _get_obs(self, state1, state2=None):
        pos_coeff = self.x_lim
        vel_coeff = self.max_velocity
        angle_coeff = np.pi
        # print ("scenario: ", self.scenario)
        # print ("state1: ", state1)
        if self.scenario == "dog_fight":
            x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, u_angular1, distance, diff_angle_1, diff_angle_2 = state1
            x2, y2, psi2, x2_diff, y2_diff, psi2_diff, x2_dot, y2_dot, u_angular2, distance, diff_angle_1, diff_angle_2 = state2

            obs_state1 = np.array([x1/pos_coeff, y1/pos_coeff, x1_diff/pos_coeff, y1_diff/pos_coeff, psi1_diff/angle_coeff, 
                                x1_dot/vel_coeff, y1_dot/vel_coeff, u_angular1, distance/(np.sqrt(2)*pos_coeff), 
                                diff_angle_1/(2*angle_coeff), diff_angle_2/(2*angle_coeff)])

            obs_state2 = np.array([x2/pos_coeff, y2/pos_coeff, x2_diff/pos_coeff, y2_diff/pos_coeff, psi2_diff/angle_coeff, 
                                x2_dot/vel_coeff, y2_dot/vel_coeff, u_angular2, distance/(np.sqrt(2)*pos_coeff), 
                                diff_angle_1/(2*angle_coeff), diff_angle_2/(2*angle_coeff)])

            return obs_state1, obs_state2
        else:
            
            x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, u_angular, distance, diff_angle_1, diff_angle_2 = state1

            obs_state1 = np.array([x1/pos_coeff, y1/pos_coeff, x1_diff/pos_coeff, y1_diff/pos_coeff, psi1_diff/angle_coeff, 
                                x1_dot/vel_coeff, y1_dot/vel_coeff, u_angular, distance/(np.sqrt(2)*pos_coeff), 
                                diff_angle_1/(2*angle_coeff), diff_angle_2/(2*angle_coeff)])

            return obs_state1

    def reset(self):
        x_ind = np.random.randint(0,5)
        y_ind = np.random.randint(0,5)
        x_lim_1 = -self.x_lim + x_ind * 20
        x_lim_2 = -self.x_lim + (x_ind + 1) * 20
        y_lim_1 = -self.y_lim + y_ind * 20
        y_lim_2 = -self.y_lim + (y_ind + 1) * 20

        # x_ind = np.random.randint(0,5)
        # y_ind = np.random.randint(0,5)
        x_lim_3 = -self.x_lim + x_ind * 20
        x_lim_4 = -self.x_lim + (x_ind + 1) * 20
        y_lim_3 = -self.y_lim + y_ind * 20
        y_lim_4 = -self.y_lim + (y_ind + 1) * 20

        x1 = self.np_random.uniform(low=x_lim_1, high=x_lim_2)
        y1 = self.np_random.uniform(low=y_lim_1, high=y_lim_2)
        psi1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        x1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        y1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        psi1_dot = self.np_random.uniform(low=-0.5, high=0.5)

        if self.scenario == "dog_fight":
            x2 = self.np_random.uniform(low=x_lim_3, high=x_lim_4)
            y2 = self.np_random.uniform(low=y_lim_3, high=y_lim_4)
            psi2 = self.np_random.uniform(low=-np.pi, high=np.pi)
            x2_dot = self.np_random.uniform(low=-0.5, high=0.5)
            y2_dot = self.np_random.uniform(low=-0.5, high=0.5)
            psi2_dot = self.np_random.uniform(low=-0.5, high=0.5)

            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            self.state1 = np.array([x1, y1, psi1, x1-x2, y1-y2, psi1-psi2, x1_dot, y1_dot, psi1_dot, distance, 0.0, 0.0])
            self.state2 = np.array([x2, y2, psi2, x2-x1, y2-y1, psi2-psi1, x2_dot, y2_dot, psi2_dot, distance, 0.0, 0.0])

            return self._get_obs(self.state1, self.state2)
        
        else:

            self.x_target = self.np_random.uniform(low=x_lim_1, high=x_lim_2)
            self.y_target = self.np_random.uniform(low=y_lim_1, high=y_lim_2)
            self.psi_target = self.np_random.uniform(low=-np.pi, high=np.pi)

            distance = np.sqrt((self.x_target - x1)**2 + (self.y_target - y1)**2)

            self.state1 = np.array([x1, y1, psi1, self.x_target - x1, self.y_target - y1, self.psi_target - psi1, x1_dot, y1_dot, psi1_dot, distance, 0., 0.])

            return self._get_obs(state1 = self.state1)


        

        

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-self.x_lim, self.x_lim, -self.y_lim, self.y_lim)

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

        if self.scenario == "dog_fight":
            self.viewer.add_onetime(self.plane2)
            self.plane2_transform.set_translation(self.state2[0], self.state2[1])
            self.plane2_transform.set_rotation(self.state2[2])
        else:
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

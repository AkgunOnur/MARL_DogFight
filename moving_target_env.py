import gym
import torch
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
from os import path
import math
import time



class MovingTarget(gym.Env):
    def __init__(self, agent_init:list, opponent_init:list, angle_init:list, map_lim=50,  agent_range=10.0, agent_angle=60.0, opponent_range=10.0, opponent_angle=60.0, name="A", opponent="B", locked_reward=1.0, get_locked_reward=-1.0, max_timesteps=500, visualization = False, seed=None):
        self.dt = .05
        self.max_angular_velocity = 1.5
        self.min_velocity = 1.0
        self.max_velocity = 12.0
        self.x_target = 0.
        self.y_target = 0.
        self.v_target = 0.0
        self.psi_target = 0.
        self.psi_goal = 0.
        self.viewer = None
        self.agent_init = agent_init
        self.opponent_init = opponent_init
        self.angle_init = angle_init
        self.dog_fight_range = agent_range
        self.agent_range = agent_angle
        self.opponent_range = opponent_range
        self.opponent_angle = opponent_angle
        self.name = name
        self.opponent = opponent
        self.state_opponent = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.map_lim = float(map_lim)
        self.locked_reward = locked_reward
        self.get_locked_reward = get_locked_reward
        self.state1 = None
        self.state2 = None
        self.name1 = name
        self.name2 = opponent
        self.reward_range = 2
        self.visualization = visualization
        self._max_episode_steps = max_timesteps
        self.timesteps = 0
        self.level = None
        self.locking_time = 80
        

        self.rockets = []
        self.rockets_transform = []
        self.fired_rockets = 0
        self.N_rockets = 5
        self.N_rockets = 50
        self.rocket_states = np.zeros((self.N_rockets, 5)) # x,y,psi,vel_lin,vel_ang
        self.rocket_loading_time = {"level3": 30, "level4":10}
        self.fire_available = False
        self.counter = 0
        self.counter_opponent = 0

        # self.metadata = {'render.modes': ['human']}

        # self.action_space = spaces.Box(
        #     low=np.array([-np.float32(self.max_angular_velocity), np.float32(self.min_velocity)]),
        #     high=np.array([np.float32(self.max_angular_velocity), np.float32(self.max_velocity)]), shape=(2,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Box(
            np.array([-self.max_angular_velocity, self.min_velocity]).astype(np.float32),
            np.array([self.max_angular_velocity, self.max_velocity]).astype(np.float32))

        high = np.array([1] * 11).astype(np.float32)

        self.observation_space = spaces.Box(low=-high, high=high)
            

        self.Seed(seed)
        self.close()
        

    def Seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        torch.manual_seed(seed)
        return [seed]


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

    def rocket_model(self, index):
        #rocket states = # x,y,psi,vel_lin,vel_ang
        target_shot = False
        x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, psi1_dot, distance, diff_angle, diff_angle_1 = self.state1 
        for i in range(index):
            vel_lin = self.rocket_states[i][3]
            vel_ang = self.rocket_states[i][4]
            self.rocket_states[i][2] = self.rocket_states[i][2] + vel_ang * self.dt
            self.rocket_states[i][2] = angle_normalize(self.rocket_states[i][2])
            x1_dot = vel_lin*np.cos(self.rocket_states[i][2] + np.pi/2)
            self.rocket_states[i][0] = self.rocket_states[i][0] + x1_dot * self.dt
            y1_dot = vel_lin*np.sin(self.rocket_states[i][2] + np.pi/2)
            self.rocket_states[i][1] = self.rocket_states[i][1] + y1_dot * self.dt

            distance = np.sqrt((x1 - self.rocket_states[i][0])**2 + (y1 - self.rocket_states[i][1])**2)
            if distance < 2.0:
                target_shot = True

        return target_shot


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
        
        start_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 - self.agent_range/2*np.pi/180)
        end_angle_1 = angle_normalize_2pi(psi1 + np.pi/2 + self.agent_range/2*np.pi/180)
        diff_angle_1 = angle_normalize_2pi(np.arctan2(self.y_target-y1, self.x_target-x1))

        start_angle_2 = angle_normalize_2pi(self.psi_target + np.pi/2 - self.opponent_angle/2*np.pi/180)
        end_angle_2 = angle_normalize_2pi(self.psi_target + np.pi/2 + self.opponent_angle/2*np.pi/180)
        diff_angle_2 = angle_normalize_2pi(np.arctan2(y1-self.y_target, x1-self.x_target))

        diff_angle = angle_normalize(psi1-self.psi_target)

        if x1 > 1.1*self.map_lim or x1 < -1.1*self.map_lim or y1 > 1.1*self.map_lim or y1 < -1.1*self.map_lim:
            reward = -100.0
            done = True
            self.counter = 0
            self.counter_opponent = 0
        elif x1 > self.map_lim or x1 < -self.map_lim or y1 > self.map_lim or y1 < -self.map_lim:
            reward = -5.0
            self.counter = 0
            self.counter_opponent = 0
            # done = True
            # print (f"\nPlane {self.name1} is out of map!")
        elif distance <= 4.0:
            reward = -25.0
            done = True
            self.counter = 0
            self.counter_opponent = 0
            # print ("Planes crashed each other!")
        else:
            # reward = np.clip(-0.4*distance / max_dist -0.6*np.abs(diff_angle_1)/(2*np.pi), -1.0, 1.0)
            reward = np.clip(-distance / (2*max_dist), -0.5, 0.5)
            if distance <= self.agent_range:
                if end_angle_1 > start_angle_1:
                    if start_angle_1 < diff_angle_1 < end_angle_1:
                        #print ("The opponent " + self.opponent + " is seen!")
                        locked = True
                        reward = self.locked_reward
                        self.counter += 1
                    else:
                        self.counter = 0
                else:
                    if diff_angle_1 > start_angle_1 or diff_angle_1 < end_angle_1:
                        #print ("The opponent " + self.opponent + " is seen!")
                        locked = True
                        reward = self.locked_reward
                        self.counter += 1
                    else:
                        self.counter = 0

            if distance <= self.opponent_range:
                if end_angle_2 > start_angle_2:
                    if start_angle_2 < diff_angle_2 < end_angle_2:
                        # print ("Mission failed!")
                        get_locked = True
                        reward = self.get_locked_reward
                        self.counter_opponent += 1
                        # if (self.level == "level3" or self.level == "level4") and self.fired_rockets < self.N_rockets and self.fire_available:
                        #     self.fire_available = False
                        #     self.rocket_states[self.fired_rockets] = np.r_[[self.x_target, self.y_target, self.psi_target], [20.0, 0]]
                        #     self.fired_rockets += 1
                    else:
                        self.counter_opponent = 0
                            
                else:
                    if diff_angle_2 > start_angle_2 or diff_angle_2 < end_angle_2:
                        # print ("Mission failed!")
                        get_locked = True
                        reward = self.get_locked_reward
                        self.counter_opponent += 1
                        # if (self.level == "level3" or self.level == "level4") and self.fired_rockets < self.N_rockets and self.fire_available:
                        #     self.fire_available = False
                        #     self.rocket_states[self.fired_rockets] = np.r_[[self.x_target, self.y_target, self.psi_target], [20.0, 0]]
                        #     self.fired_rockets += 1
                    else:
                        self.counter_opponent = 0
                            

            # if locked and get_locked and self.level != "level4":
            #     reward = 0.0
        
        # if self.fire_available == False: # fire power loads up in rocket loading time
        #     self.fire_counter += 1

        # if self.fire_available == False and self.fire_counter >= self.rocket_loading_time[self.level]: # if you wait enough, you can fire again
        #     self.fire_available = True
        #     self.fire_counter = 0

        if self.timesteps >= self._max_episode_steps:
            done = True
        
        if self.counter >= self.locking_time and self.counter_opponent >= self.locking_time :
            done = True
            reward = -25.0
            # print("They destroyed each other!")
        elif self.counter_opponent >= self.locking_time:
            done = True
            reward = -100.0
            # print("Enemy destroyed the agent!")
        elif self.counter >= self.locking_time :
            done = True
            reward = 100.0
            # print("Agent destroyed the enemy!")


        # To restrain the position of plane
        # x1 = np.clip(x1, -self.map_lim, self.map_lim)
        # y1 = np.clip(y1, -self.map_lim, self.map_lim)

        # if self.level == "level3" or self.level == "level4":
        #     target_shot = self.rocket_model(self.fired_rockets)
        #     if target_shot:
        #         done = True
        #         reward = -1

        if self.visualization:
            self.render()  
            # time.sleep(5.0)
            if done:
                self.close()

        self.state1 = (x1, y1, psi1, x1_diff, y1_diff, psi1_diff, x1_dot, y1_dot, u_angular1, distance, diff_angle, diff_angle_1)


        if self.timesteps % 50 == 0:
            self.v_target = self.np_random.uniform(low=6.0, high=10.0)
            self.psi_target += self.np_random.uniform(low=-np.pi/4, high=np.pi/4)

        # if (self.level == "level2" or self.level == "level3")  and self.timesteps % 50 == 0:
        #     self.v_target = np.random.uniform(low=4.0, high=8.0)
        #     self.psi_target += np.random.uniform(low=-np.pi/4, high=np.pi/4)
        # elif (self.level == "level4" or self.level == "level5")  and self.timesteps % 25 == 0:
        #     self.v_target = np.random.uniform(low=6.0, high=10.0)
        #     self.psi_target += np.random.uniform(low=-np.pi/2, high=np.pi/2)
            

        # print (f"reward: {reward:.4} linear: {u_linear1:.4} angular: {u_angular1:.4}")
        # time.sleep(0.25)
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
        self.counter = 0
        self.counter_opponent = 0

        # x1 = self.np_random.uniform(low=self.agent_init[0], high=self.agent_init[1])
        # y1 = self.np_random.uniform(low=self.agent_init[2], high=self.agent_init[3])
        # self.x_target = self.np_random.uniform(low=self.opponent_init[0], high=self.opponent_init[1])
        # self.y_target = self.np_random.uniform(low=self.opponent_init[2], high=self.opponent_init[3])

        # x1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        # y1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        # psi1_dot = self.np_random.uniform(low=-0.5, high=0.5)
        # self.psi_target = self.np_random.uniform(low=-np.pi, high=np.pi)
        # psi1 = self.np_random.uniform(low=-np.pi, high=np.pi)
        # self.v_target = self.np_random.uniform(low=6.0, high=10)

        x1 = self.agent_init[0]
        y1 = self.agent_init[1]
        self.x_target = self.opponent_init[0]
        self.y_target = self.opponent_init[1]

        x1_dot = 0.25
        y1_dot = 0.25
        psi1_dot = 0.2
        self.v_target = 10.0
        psi1 = self.angle_init[0]
        self.psi_target = self.angle_init[1]


        # if self.level == "level1":
        #     x1 = self.np_random.uniform(low=-self.map_lim / 4.0, high=self.map_lim / 4.0)
        #     y1 = self.np_random.uniform(low=-self.map_lim / 4.0, high=self.map_lim / 4.0)
        #     self.x_target = self.np_random.uniform(low=-self.map_lim / 4.0, high=self.map_lim / 4.0)
        #     self.y_target = self.np_random.uniform(low=-self.map_lim / 4.0, high=self.map_lim / 4.0)
        # elif self.level == "level2":
        #     x1 = self.np_random.uniform(low=-self.map_lim / 2.0, high=self.map_lim / 2.0)
        #     y1 = self.np_random.uniform(low=-self.map_lim / 2.0, high=self.map_lim / 2.0)
        #     self.x_target = self.np_random.uniform(low=-self.map_lim / 2.0, high=self.map_lim / 2.0)
        #     self.y_target = self.np_random.uniform(low=-self.map_lim / 2.0, high=self.map_lim / 2.0)
        # elif self.level == "level3":
        #     x1 = self.np_random.uniform(low=-self.map_lim + 10.0,  high=-self.map_lim + 20.0)
        #     y1 = self.np_random.uniform(low=-self.map_lim + 20.0, high=-self.map_lim + 40.0)
        #     self.x_target = self.np_random.uniform(low=self.map_lim - 20.0, high=self.map_lim - 10.0)
        #     self.y_target = self.np_random.uniform(low=self.map_lim - 40.0, high=self.map_lim - 20.0)
            
        # elif self.level == "level4":
        #     x1 = self.np_random.uniform(low=-self.map_lim, high=-self.map_lim + 10.0)
        #     y1 = self.np_random.uniform(low=-self.map_lim, high=-self.map_lim + 20.0)
        #     self.x_target = self.np_random.uniform(low=self.map_lim - 10.0, high=self.map_lim)
        #     self.y_target = self.np_random.uniform(low=self.map_lim - 20.0, high=self.map_lim)
        # elif self.level == "level5":
        #     x1 = self.np_random.uniform(low=-self.map_lim,  high=-self.map_lim + 5.0)
        #     y1 = self.np_random.uniform(low=-self.map_lim, high=-self.map_lim + 5.0)
        #     self.x_target = self.np_random.uniform(low=self.map_lim - 5.0, high=self.map_lim)
        #     self.y_target = self.np_random.uniform(low=self.map_lim - 5.0, high=self.map_lim)


        # if self.level == "level2" or self.level == "level3":
        #     self.psi_target = self.np_random.uniform(low=np.pi - np.pi/2, high=np.pi)
        #     psi1 = self.psi_target - np.pi #self.np_random.uniform(low=-np.pi, high=np.pi)
        # elif self.level == "level1" or self.level == "level4" or self.level == "level5":
        #     self.psi_target = self.np_random.uniform(low=-np.pi, high=np.pi)
        #     psi1 = self.np_random.uniform(low=-np.pi, high=np.pi)


        self.fire_counter = 0
        self.fired_rockets = 0
        self.rockets = []
        self.rockets_transform = []
        self.rocket_states = np.zeros((self.N_rockets, 5))
        self.fire_available = False
        

        

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
            
            fname1 = path.join(path.dirname(__file__), "assets/gplane.png")
            fname2 = path.join(path.dirname(__file__), "assets/plane2.png")
            fname3 = path.join(path.dirname(__file__), "assets/rocket3.png")
            self.plane1 = rendering.Image(fname1, 7., 7.)
            self.plane2 = rendering.Image(fname2, 7., 7.)
            #self.imgtrans = rendering.Transform()

            for i in range(self.N_rockets):
                self.rockets_transform.append(rendering.Transform())
                self.rockets.append(rendering.Image(fname3, 3., 6.))
                self.rockets[i].add_attr(self.rockets_transform[i])

            self.plane1.add_attr(self.plane1_transform)
            self.plane2.add_attr(self.plane2_transform)

        self.viewer.add_onetime(self.plane1)
        self.plane1_transform.set_translation(self.state1[0], self.state1[1])
        self.plane1_transform.set_rotation(self.state1[2])

        for i in range(self.fired_rockets):
            self.viewer.add_onetime(self.rockets[i])
            self.rockets_transform[i].set_translation(self.rocket_states[i][0], self.rocket_states[i][1])
            self.rockets_transform[i].set_rotation(self.rocket_states[i][2])

        
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
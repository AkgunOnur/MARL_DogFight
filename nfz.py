import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from gym.envs.classic_control import rendering
import pprint
import time

################################################################################
class Zones():
    def __init__(self, numberOfNFZ):
        self.Dict = {}
        self.numberOfNFZ = numberOfNFZ
        k = 0
        while k < numberOfNFZ:
            # dynamically create key
            zoneID = str("Zone" + str(k+1))
            # calculate value
            dummyDict = {"zx" : 0.0, "zy": 0.0, "zr": 0.0}
            self.Dict[zoneID] = dummyDict 
            k += 1
################################################################################
class Rockets():
    def __init__(self, numberOfRockets):
        self.numberOfRockets = numberOfRockets
        self.Dict = {}
        k = 0
        while k < numberOfRockets:
            # dynamically create key
            rocketID = str("Rocket" + str(k+1))
            # calculate value
            dummyDict = {"rx" : 0.0, "ry": 0.0, "rv": 0.0,
                        "rpsi": 0.0, "rpsidot": 0.0}
            self.Dict[rocketID] = dummyDict 
            k += 1


################################################################################
class NFZone(gym.Env):
    def __init__(self, numberOfNFZ=2):
        super(NFZone, self).__init__()
        self.seed()

        #Reset transforms
        self.nfz = []
        self.nfz_transform = []
        self.rockets = []
        self.rocket_transform = []

        self.inside = False
        self.dt = .05

        #Number of Rockets and Zones
        self.numberOfNFZ = numberOfNFZ
        self.numberOfRockets = self.numberOfNFZ

        #Observation and action spaces
        #STATES: [relative x, relative y, agent_heading-angle_between] - Target
        #      [relative x, relative y, agent_heading-angle_between, zone_radius] - Zone1
        #      [relative x, relative y, agent_heading-angle_between, zone_radius] - Zone2
        #      [relative x, relative y, agent_heading-angle_between, rocket_heading-angle_between, agent_vel - rock-vel, fired_status] - Rocket1
        #      [relative x, relative y, agent_heading-angle_between, rocket_heading-angle_between, agent_vel - rock-vel, fired_status] - Rocket2

        High = np.array([100., 100., np.pi, 100., 100., np.pi, 40.0, 100., 100., np.pi, 40.0, 100., 100., np.pi, np.pi, 5.0, 1.0, 100., 100., np.pi, np.pi, 5.0, 1.0], dtype=np.float32)
        Low = np.array([-100., -100., -np.pi, -100., -100., -np.pi, 15.0, -100., -100., -np.pi, 15.0, -100., -100., -np.pi, -np.pi, -1.0, 0.0, -100., -100., -np.pi, -np.pi, -1.0, 0.0], dtype=np.float32)
        #print("SHAPE HIGH: ", high.shape)

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1,  1]), shape=(2,),
            dtype=np.float32
        ) #Map max/min into vel[1.0, 5.0], psidot[-np.pi/8, np.pi/8]

        self.observation_space = spaces.Box(
            low=Low,
            high=High,
            shape=(23,),
            dtype=np.float32
        )
        self.viewer = None

        # agent and target parameters
        self.agent = {}
        self.agent["v"] = 2
        self.agent["x"] = 0
        self.agent["y"] = 0
        self.agent["vx"] = 0
        self.agent["vy"] = 0
        self.agent["psi"] = 0
        self.agent["psi_dot"] = 0
        self.agent["angle_between"] = 0

        #Other Episodic parameters
        self.target = {}
        self.target["x"] = 0
        self.target["y"] = 0
        self.target["psi"] = 0

        self.reward= 0
        self.done = False

        #Initialize Zones and Rockets        
        self.Z = Zones(self.numberOfNFZ)
        self.R = Rockets(self.numberOfRockets)
    ########################################
    def checkCloseness(self, zone, unit):
        for i in range(self.numberOfNFZ):
            dist = np.sqrt((unit["x"]-zone["Zone" + str(i+1)]["zx"])**2 + (unit["y"]-zone["Zone" + str(i+1)]["zy"])**2)
            if dist <= zone["Zone" + str(i+1)]["zr"]/2:
                return True
            else:
                continue
        return False
    ########################################
    def initializeZones(self):
        k = 0
        while k < self.numberOfNFZ:
            # dynamically create key
            zoneID = str("Zone" + str(k+1))
            # calculate value
            dummyDict = {"zx" : round(np.random.uniform(low=-50.0, high=50.0),2), "zy": round(np.random.uniform(low=-50.0, high=50.0),2), "zr": round(np.random.uniform(low=15.0, high=40.0),2)}
            self.Z.Dict[zoneID] = dummyDict 
            k += 1
    ########################################
    def initializeTarget(self, zone):
        tooClose = True
        while tooClose:
            self.target["x"] = np.random.uniform(low=-50.0, high=50.0)
            self.target["y"] = np.random.uniform(low=-50.0, high=50.0)
            tooClose = self.checkCloseness(zone.Dict, self.target)
            if tooClose:
                continue
            else:
                break
    ########################################
    def initializeAgent(self, zone, target):
        tooClose = True
        while tooClose:
            self.agent["x"] = np.random.uniform(low=-45.0, high=45.0)
            self.agent["y"] = np.random.uniform(low=-45.0, high=45.0)
            tooClose = self.checkCloseness(zone.Dict, self.agent)
            if tooClose:
                continue
            else:
                break
        self.agent["angle_between"] =  angle_normalize(np.arctan2(self.agent["y"]-target["y"], self.agent["x"]-target["x"]))
        self.agent["psi"] = np.random.uniform(low=-np.pi, high=np.pi)
        self.agent["psi_dot"] = 0.0
        self.agent["v"] = np.random.uniform(low=1.0, high=5.0)
        self.agent["vx"] = self.agent["v"] * np.cos(self.agent["psi"])
        self.agent["vy"] = self.agent["v"] * np.sin(self.agent["psi"])
    ########################################
    def initializeRockets(self, zone, agent):
        k = 0
        while k < self.numberOfRockets:
            # dynamically create key
            rocketID = str("Rocket" + str(k+1))
            # calculate value
            dummyDict = {"rx" : zone.Dict["Zone"+str(k+1)]["zx"], "ry": zone.Dict["Zone"+str(k+1)]["zy"], "rv": 0.0,
                        "rpsi": angle_normalize(np.arctan2(agent["y"]-zone.Dict["Zone"+str(k+1)]["zy"] , agent["x"]-zone.Dict["Zone"+str(k+1)]["zx"])), "fired": 0.0}
            self.R.Dict[rocketID] = dummyDict 
            k += 1
    ########################################
    def UpdateRockets(self):
        k = 0
        #Check if agent inside any NF Zone
        self.inside = self.checkCloseness(self.Z.Dict, self.agent)
        
        while k < self.numberOfRockets:
            # dynamically create key
            rocketID = str("Rocket" + str(k+1))

            #if outside of the map respawn rocket and set fired status and velocity zero
            if abs(self.R.Dict[rocketID]["rx"])>50.0 or abs(self.R.Dict[rocketID]["ry"])>50.0:
                self.inside = False
                self.R.Dict[rocketID]["rv"] = 0.0
                self.R.Dict[rocketID]["fired"] = 0.0
                self.R.Dict[rocketID]["rx"] = self.Z.Dict["Zone"+str(k+1)]["zx"]
                self.R.Dict[rocketID]["ry"] = self.Z.Dict["Zone"+str(k+1)]["zy"]
                self.R.Dict[rocketID]["rpsi"] = angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket"+str(k+1)]["ry"] , self.agent["x"]-self.R.Dict["Rocket"+str(k+1)]["rx"]))

            #Fire rockets if not already fired
            elif(self.inside) or self.R.Dict[rocketID]["fired"] == 1.0:
                self.R.Dict[rocketID]["rv"] = 3.0
                self.R.Dict[rocketID]["fired"] = 1.0

                #ad =  angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket"+str(k+1)]["ry"] , self.agent["x"]-self.R.Dict["Rocket"+str(k+1)]["rx"]))
                #self.R.Dict[rocketID]["rpsi"] = ad#self.R.Dict[rocketID]["rpsi"] - np.sign(ad)*np.pi/12*self.dt 
                #self.R.Dict[rocketID]["rpsi"] = angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket"+str(k+1)]["ry"], self.agent["x"]-self.R.Dict["Rocket"+str(k+1)]["rx"]))
                self.R.Dict[rocketID]["rx"] = self.R.Dict[rocketID]["rx"] + self.R.Dict[rocketID]["rv"] * np.cos(self.R.Dict[rocketID]["rpsi"])*self.dt
                self.R.Dict[rocketID]["ry"] = self.R.Dict[rocketID]["ry"] + self.R.Dict[rocketID]["rv"] * np.sin(self.R.Dict[rocketID]["rpsi"])*self.dt
            else:
                self.R.Dict[rocketID]["rpsi"] =  angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket"+str(k+1)]["ry"] , self.agent["x"]-self.R.Dict["Rocket"+str(k+1)]["rx"])) 
            k += 1
    ########################################
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    ########################################

    def step(self, action):
        action = np.clip(action, [-1.,-1.],
                    [1.,1.])

        #Scale  and update agent actions
        self.agent["v"] = action[0]*2 + 3 #v linear - action scaled between [1,5]
        self.agent["psi_dot"] = action[1]*(np.pi/8) #psi_dot - action scaled between [-pi/8, pi/8]
        
        #Update agent parameters
        self.agent["psi"] = angle_normalize(self.agent["psi"] + self.agent["psi_dot"]*self.dt)
        self.agent["vx"] = self.agent["v"] * np.cos(self.agent["psi"])
        self.agent["vy"] = self.agent["v"] * np.sin(self.agent["psi"])

        self.agent["x"] += self.agent["vx"] * self.dt
        self.agent["y"] += self.agent["vy"] * self.dt
        self.agent["angle_between"] =  angle_normalize(np.arctan2(self.agent["y"]-self.target["y"], self.agent["x"]-self.target["x"]))
        
        self.UpdateRockets()

        
        #Get state
        self.state = np.array([self.agent["x"]-self.target["x"], self.agent["y"]-self.target["y"], angle_normalize(self.agent["psi"]-self.agent["angle_between"]),
                               self.agent["x"]-self.Z.Dict["Zone1"]["zx"], self.agent["y"]-self.Z.Dict["Zone1"]["zy"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.Z.Dict["Zone1"]["zy"], self.agent["x"]-self.Z.Dict["Zone1"]["zx"]))), self.Z.Dict["Zone1"]["zr"],
                               self.agent["x"]-self.Z.Dict["Zone2"]["zx"], self.agent["y"]-self.Z.Dict["Zone2"]["zy"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.Z.Dict["Zone2"]["zy"], self.agent["x"]-self.Z.Dict["Zone2"]["zx"]))), self.Z.Dict["Zone2"]["zr"],
                               self.agent["x"]-self.R.Dict["Rocket1"]["rx"], self.agent["y"]-self.R.Dict["Rocket1"]["ry"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket1"]["ry"], self.agent["x"]-self.R.Dict["Rocket1"]["rx"]))),
                               angle_normalize(self.R.Dict["Rocket1"]["rpsi"]-angle_normalize(np.arctan2(self.R.Dict["Rocket1"]["ry"]-self.agent["y"], self.R.Dict["Rocket1"]["rx"]-self.agent["x"]))), self.agent["v"]-self.R.Dict["Rocket1"]["rv"], self.R.Dict["Rocket1"]["fired"],
                               self.agent["x"]-self.R.Dict["Rocket2"]["rx"], self.agent["y"]-self.R.Dict["Rocket2"]["ry"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket2"]["ry"], self.agent["x"]-self.R.Dict["Rocket2"]["rx"]))),
                               angle_normalize(self.R.Dict["Rocket2"]["rpsi"]-angle_normalize(np.arctan2(self.R.Dict["Rocket2"]["ry"]-self.agent["y"], self.R.Dict["Rocket2"]["rx"]-self.agent["x"]))),self.agent["v"]-self.R.Dict["Rocket2"]["rv"], self.R.Dict["Rocket2"]["fired"]],dtype=np.float32)

        #STATES: [relative x, relative y, agent_heading-angle_between] - Target
        #      [relative x, relative y, agent_heading-angle_between, zone_radius] - Zone1
        #      [relative x, relative y, agent_heading-angle_between, zone_radius] - Zone2
        #      [relative x, relative y, agent_heading-angle_between, rocket_heading-angle_between, agent_vel - rock-vel, fired_status] - Rocket1
        #      [relative x, relative y, agent_heading-angle_between, rocket_heading-angle_between, agent_vel - rock-vel, fired_status] - Rocket2

        #Default reward
        self.reward =  -np.sqrt((self.state[0])**2 + (self.state[1])** 2) - 10 * np.abs(self.state[2])
        
        #Going outside map
        if(abs(self.agent["x"]) >= 50 or abs(self.agent["y"]) >= 50):
            self.done = True
            self.reward = -500
        #Hit by rocket
        elif(np.sqrt((self.state[11])**2 + (self.state[12]) ** 2) <= 3.0 or np.sqrt((self.state[17])**2 + (self.state[18]) ** 2) <= 3.0):
            self.done = True
            self.reward = -200
        #Reaching target
        elif(np.sqrt((self.state[0])**2 + (self.state[1]) ** 2) <= 3.0):
            self.done = True
            self.reward = 2000
        #Getting in NFZ    
        elif((np.sqrt(self.state[3]**2 + self.state[4] ** 2) <= self.state[6]/2) or (np.sqrt(self.state[7]**2 + self.state[8] ** 2) <= self.state[10]/2)):
            self.reward = -200
        return self.state, self.reward, self.done, {}

    ########################################
    def reset(self):
        #Reset zone parameters
        self.initializeZones()

        #Reset target location
        self.initializeTarget(self.Z)

        #Reset agent parameters
        self.initializeAgent(self.Z, self.target)

        #Reset rocket parameters
        self.initializeRockets(self.Z, self.agent)

        # #Reset transforms
        self.nfz = []
        self.nfz_transform = []
        self.rockets = []
        self.rocket_transform = []
        self.inside = False
        
        #STATES: [relative x, relative y, agent_heading-angle_between] - Target
        #      [relative x, relative y, agent_heading-angle_between, zone_radius] - Zone1
        #      [relative x, relative y, agent_heading-angle_between, zone_radius] - Zone2
        #      [relative x, relative y, agent_heading-angle_between, rocket_heading-angle_between, agent_vel - rock-vel, fired_status] - Rocket1
        #      [relative x, relative y, agent_heading-angle_between, rocket_heading-angle_between, agent_vel - rock-vel, fired_status] - Rocket2

        self.state = np.array([self.agent["x"]-self.target["x"], self.agent["y"]-self.target["y"], angle_normalize(self.agent["psi"]-self.agent["angle_between"]),
                               self.agent["x"]-self.Z.Dict["Zone1"]["zx"], self.agent["y"]-self.Z.Dict["Zone1"]["zy"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.Z.Dict["Zone1"]["zy"], self.agent["x"]-self.Z.Dict["Zone1"]["zx"]))), self.Z.Dict["Zone1"]["zr"],
                               self.agent["x"]-self.Z.Dict["Zone2"]["zx"], self.agent["y"]-self.Z.Dict["Zone2"]["zy"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.Z.Dict["Zone2"]["zy"], self.agent["x"]-self.Z.Dict["Zone2"]["zx"]))), self.Z.Dict["Zone2"]["zr"],
                               self.agent["x"]-self.R.Dict["Rocket1"]["rx"], self.agent["y"]-self.R.Dict["Rocket1"]["ry"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket1"]["ry"], self.agent["x"]-self.R.Dict["Rocket1"]["rx"]))),
                               angle_normalize(self.R.Dict["Rocket1"]["rpsi"]-angle_normalize(np.arctan2(self.R.Dict["Rocket1"]["ry"]-self.agent["y"], self.R.Dict["Rocket1"]["rx"]-self.agent["x"]))), self.agent["v"]-self.R.Dict["Rocket1"]["rv"], self.R.Dict["Rocket1"]["fired"],
                               self.agent["x"]-self.R.Dict["Rocket2"]["rx"], self.agent["y"]-self.R.Dict["Rocket2"]["ry"], angle_normalize(self.agent["psi"]-angle_normalize(np.arctan2(self.agent["y"]-self.R.Dict["Rocket2"]["ry"], self.agent["x"]-self.R.Dict["Rocket2"]["rx"]))),
                               angle_normalize(self.R.Dict["Rocket2"]["rpsi"]-angle_normalize(np.arctan2(self.R.Dict["Rocket2"]["ry"]-self.agent["y"], self.R.Dict["Rocket2"]["rx"]-self.agent["x"]))),self.agent["v"]-self.R.Dict["Rocket2"]["rv"], self.R.Dict["Rocket2"]["fired"]],dtype=np.float32)

        self.done = False
        return self.state
    ########################################
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-50, 50, -50, 50)

            # Add plane pic
            fname = "./assets/plane.png"
            self.plane_transform = rendering.Transform()
            self.plane = rendering.Image(fname, 5., 5.)
            self.plane.add_attr(self.plane_transform)

            # Added zone pic
            fname2 = "./assets/target.png"
            self.target_transform = rendering.Transform()
            self.target_img = rendering.Image(fname2, 5., 5.)
            self.target_img.add_attr(self.target_transform)

        if not self.rockets:
            # Added zone pic
            fname3 = "./assets/rocket.png"
            for i in range(self.numberOfRockets):
                self.rocket_transform.append(rendering.Transform())
                self.rockets.append(rendering.Image(fname3, 5., 5.))
                self.rockets[i].add_attr(self.rocket_transform[i])
            
        if not self.nfz:  
            # Added zone pic
            fname4 = "./assets/target.png"
            for i in range(self.numberOfNFZ):
                self.nfz_transform.append(rendering.Transform())
                self.nfz.append(rendering.Image(fname4, self.Z.Dict["Zone"+str(i+1)]["zr"], self.Z.Dict["Zone"+str(i+1)]["zr"]))
                self.nfz[i].add_attr(self.nfz_transform[i])

        # Add plane position and orientation
        self.viewer.add_onetime(self.plane)
        self.plane_transform.set_translation(self.agent["x"], self.agent["y"])
        self.plane_transform.set_rotation(self.agent["psi"])

        # Add target position and orientation
        self.viewer.add_onetime(self.target_img)
        self.target_transform.set_translation(self.target["x"], self.target["y"])
        self.target_transform.set_rotation(self.target["psi"])

        for i in range(self.numberOfNFZ):
            # Add nfz position and orientation
            self.viewer.add_onetime(self.nfz[i])
            self.nfz_transform[i].set_translation(self.Z.Dict["Zone"+str(i+1)]["zx"], self.Z.Dict["Zone"+str(i+1)]["zy"])
            self.nfz_transform[i].set_rotation(0.0)
            # self.nfz_transform[i].set_scale(self.Z.Dict["Zone"+str(i+1)]["zr"], self.Z.Dict["Zone"+str(i+1)]["zr"])

        for i in range(self.numberOfRockets):
            # Add target position and orientation
            self.viewer.add_onetime(self.rockets[i])
            self.rocket_transform[i].set_translation(self.R.Dict["Rocket"+str(i+1)]["rx"], self.R.Dict["Rocket"+str(i+1)]["ry"])
            self.rocket_transform[i].set_rotation(self.R.Dict["Rocket"+str(i+1)]["rpsi"])


        return self.viewer.render(return_rgb_array=mode == 'human')

    ########################################
    def close(self):
        pass

################################################################################
def angle_normalize(angle):
    while(angle >= np.pi):
        angle -= 2 * np.pi
    while(angle < -np.pi):
        angle += 2 * np.pi
    return angle
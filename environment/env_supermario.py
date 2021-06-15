"""An OpenAI Gym interface to the NES game <TODO: Game Name>"""
from nes_py import NESEnv
import os
import numpy as np

RAM_MAP = {
    'Lives' : 0x075A,
    'Coins' : 0x075E,
    'World' : 0x075F,
    'Level' : 0x0760,
    'Timer' : (0x07F8,0x07FA),
    'Horizontal' : 0x006D,
    'Vertical' : 0x00B5,
    'X' : 0x0086,
    'State' : 0x000E,
    'FloatState' : 0x001D,
    'Mode' : 0x0770,
    'Vertical' : 0x00B5,
    'Y' : 0x00CE,
    'Tile' :  (0x0500,0x069F),
    'X_Pos_Scroll':0x0755, #==0x03AD?
    'Enemy_Drawn' : (0x000F,0x0013),
    'Enemy_Pos' : (0x04B0,0x04C4)
}
PIXEL_WIDTH = 16
PIXEL_HEIGHT = 16
#https://pypi.org/project/nes-py/
#https://datacrystal.romhacking.net/wiki/Super_Mario_Bros.:RAM_map

class SuperMarioEnv(NESEnv):
    """An OpenAI Gym interface to the NES game <TODO: Game Name>"""

    def __init__(self):
        """Initialize a new <TODO: Game Name> environment."""
        path_ROM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "super-mario-bros.nes")
        super(SuperMarioEnv, self).__init__(path_ROM)
        # setup any variables to use in the below callbacks here
        self.prev_time = 0
        self.prev_x = 0
        self.screen_offset = 0
        self.screen_idx = 0
        self.gap = 0
        self.reset()
        self.make_single_stage()
        self._skip_start_screen()
        #self.change_timer(2,0,0)
        self.make_life_once()
        self._backup()

    def step(self,action):
        _, reward, done, info = super().step(action)
        observation = self.simple_observation()
        return observation, reward, done, info

    def reset(self):
        super().reset()
        observation = self.simple_observation()
        return observation

    def simple_observation(self):
        observation = self.ram[RAM_MAP['Tile'][0]:RAM_MAP['Tile'][1]+1]
        observation = observation.reshape(2,13,16)
        observation = np.concatenate((observation[0],observation[1]),axis=1)
        if self.screen_offset < 16:
            observation = observation[0:13,self.screen_offset:self.screen_offset+16]
        else:
            lobs = observation[0:13,self.screen_offset:32]
            robs = observation[0:13,0:self.screen_offset-16]
            observation = np.concatenate((lobs,robs),axis=1)

        observation = np.where(observation>0, 255, 0)

        enemy_drawn = self.ram[RAM_MAP['Enemy_Drawn'][0]:RAM_MAP['Enemy_Drawn'][1]+1]
        enemy_pos = self.ram[RAM_MAP['Enemy_Pos'][0]:RAM_MAP['Enemy_Pos'][1]]
        for i in range(5):
            if enemy_drawn[i] == 1:
                x = (enemy_pos[i*4]+self.gap+PIXEL_WIDTH//2)//PIXEL_WIDTH
                y = (enemy_pos[i*4+1])//PIXEL_HEIGHT-2
                if y<0 or y>=13 or x<0 or x>=16:
                    continue
                else :
                    observation[y][x] = 85
        mario_x = (self.screen_x+self.gap+PIXEL_WIDTH//2)//PIXEL_WIDTH
        mario_y = (self.cur_y-1)//PIXEL_HEIGHT
        if mario_y >=0 and mario_y< 13:
            observation[mario_y][mario_x] = 190
        return observation

    def _skip_start_screen(self):
        while self.cur_time >= self.prev_time:
            self.prev_time = self.cur_time
            self._frame_advance(8)
            self._frame_advance(0)
    def make_life_once(self):
        self.write_byte(RAM_MAP['Lives'],1)

    def make_single_stage(self):
        self.write_byte(RAM_MAP['World'],0)
        self.write_byte(RAM_MAP['Level'],0)

#    def change_timer(self,h,t,o):
#        self.write_byte(RAM_MAP['Timer'][0],h)
#        self.write_byte(RAM_MAP['Timer'][0]+1,t)
#        self.write_byte(RAM_MAP['Timer'][0]+2,o)

    def _will_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        self.prev_time = 0
        self.prev_x = 0
        # use this method to perform setup before and episode resets.
        # the method returns None

    def _did_reset(self):
        """Handle any RAM hacking after a reset occurs."""
        # use this method to access the RAM of the emulator 
        # and perform setup for each episode.
        # the method returns None
        self.prev_time = self.cur_time
        self.prev_x = self.cur_x
        self.screen_offset = 0

    def _did_step(self, done):
        """
        Handle any RAM hacking after a step occurs.

        Args:
            done: whether the done flag is set to true

        Returns:
            None

        """
        self.prev_x = self.cur_x
        self.prev_time = self.cur_time
        self.screen_idx = (self.cur_x-self.screen_x)//PIXEL_WIDTH
        self.screen_offset = self.screen_idx%32
        self.gap = self.cur_x - (self.screen_x+self.screen_idx*PIXEL_WIDTH)
        if done:
            return
        if self.read_byte(RAM_MAP['State']) == 0x000B or self.read_byte(RAM_MAP['Vertical']) > 1:
            self.write_byte([RAM_MAP['State']],0x0006)
            self._frame_advance(0)

    def _get_reward(self):
        """Return the reward after a step occurs."""
        x_reward = (self.cur_x - self.prev_x)#*((self.cur_x//1000)+1)
        reward = x_reward + self.is_complete * 10u
        penalty = self.cur_time - self.prev_time
        penalty += self.is_dead*-10
        return reward - penalty

    def _get_done(self):
        """Return True if the episode is over, False otherwise."""
        return self.is_dead or self.is_complete

    def _get_info(self):
        """Return the info after a step occurs."""
        return {
            'time' : self.cur_time,
            'x' : self.cur_x,
            'y' : self.cur_y,
            'dead' : self.is_dead
        }

    def read_byte(self, address):
        return self.ram[address]

    def read_bytes(self, start_address, end_address):
        return int("".join(map(str,self.ram[start_address:end_address+1])))

    def write_byte(self, address, value):
        self.ram[address] = value

    @property
    def screen_x(self):
        return self.read_byte(RAM_MAP['X_Pos_Scroll'])
    @property
    def cur_time(self):
        return self.read_bytes(RAM_MAP['Timer'][0],RAM_MAP['Timer'][1])
    @property
    def cur_x(self):
        return self.read_byte(RAM_MAP['Horizontal'])*0x100 + self.read_byte((RAM_MAP['X']))
    @property
    def cur_y(self):
        return self.read_byte(RAM_MAP['Y'])
    @property
    def is_complete(self):
        return self.read_byte(RAM_MAP['Mode'])==0x0002 or self.read_byte(RAM_MAP['FloatState'])==0x0003
    @property
    def is_dead(self):
        return self.read_byte(RAM_MAP['State'])==0x000B or self.read_byte(RAM_MAP['State'])== 0x0006


# explicitly define the outward facing API for the module
__all__ = [SuperMarioEnv.__name__]

"""
Reset
The reset lifecycle executes in order like this pseudocode:

_will_reset()
reset()
_did_reset()
obs = screen
return obs

Step
The step lifecycle executes in order like this pseudocode:

reward = 0
done = False
info = {}
for _ in range(frameskip):
    step()
    reward += _get_reward()
    done = done or _get_done()
    info = _get_info()
_did_step()
obs = screen
return obs, reward, done, info

RAM
The RAM behaves like any other NumPy vector.

Read Byte
self.ram[address]

Write Byte
self.ram[address] = value

Frame Advance
self._frame_advance(action)

Create Backup State
self._backup()
"""
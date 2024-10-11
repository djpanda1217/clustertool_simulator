import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import numpy as np
import EQP_Scheduler as scd
import logging
import random

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='eqp-scheduler-v0',                                   # call it whatever you want
    entry_point='EQP_Scheduler_env:EQP_Scheduler_Env',       # module_name:class_name
)

MAX_STEP_LIMIT = 10000
MAX_WAFER_NO = 1000
MAX_CHAMBER_COUNT = 6
MAX_PROCESSED_WAFER_WAIT_TIME = 100 # no use

VTR_EXECUTION_TIME = 1
CMB_MIN_EXECUTION_TIME = 12
CMB_MAX_EXECUTION_TIME = 13
VAC_ATM_CHANGE_TIME = 3
CMD_CLEANING_LIMIT = 20
CMD_CLEANING_TIME = 3
ATR_EXECUTION_TIME = 1
ALIENER_EXECUTION_TIME = 1
LP_IN_OUT_PROCESS_TIME = 1

class EQP_Scheduler_Env(gym.Env):

    metadata = {"render_modes": ["human"], 'render_fps': 1}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.scd = scd.EQP_Scheduler(MAX_WAFER_NO, VTR_EXECUTION_TIME,  
                                     CMB_MIN_EXECUTION_TIME, CMB_MAX_EXECUTION_TIME, VAC_ATM_CHANGE_TIME, 
                                     CMD_CLEANING_LIMIT, CMD_CLEANING_TIME, ATR_EXECUTION_TIME, 
                                     ALIENER_EXECUTION_TIME, LP_IN_OUT_PROCESS_TIME)
        
        self.action_space = spaces.Tuple((
            spaces.Discrete(len(scd.VTRAction)),
            spaces.Discrete(len(scd.ATRAction))
        ))

        self.observation_space = spaces.Box(
            low=0,
            high=np.array([
                1,  # Entry wafer present
                1,  # Exit wafer present
                1,  # VTR Left arm wafer present
                1,  # VTR Right arm wafer present
                *[1 for _ in range(MAX_CHAMBER_COUNT)],  # Chamber wafers present (6 chambers)
                1,  # ATR Left arm wafer present
                1,  # ATR Right arm wafer present
                1,  # Aligner wafer present
                1,  # LP_IN wafer present
                1,  # LP_OUT wafer present
                2,  # Entry wafer process state
                2,  # Exit wafer process state
                2,  # VTR Left arm wafer process state
                2,  # VTR Right arm wafer process state
                *[2 for _ in range(MAX_CHAMBER_COUNT)],  # Chamber wafers process states (6 chambers)
                2,  # ATR Left arm wafer process state
                2,  # ATR Right arm wafer process state
                2,  # Aligner wafer process state
                2,  # LP_IN wafer process state
                2,  # LP_OUT wafer process state
                1,  # VTR Left arm available
                1,  # VTR Right arm available
                *[1 for _ in range(MAX_CHAMBER_COUNT)],  # Chamber available (6 chambers)
                1,  # ATR Left arm available
                1,  # ATR Right arm available
                1,  # Aligner available
                1,  # LP_IN available
                1,  # LP_OUT available
                1,  # Entry vac_or_atm
                1,  # Exit vac_or_atm
            ]),
            dtype=np.float32
        )

        self.reset_cnt = 0
        self.cumulative_reward = 0
        self.step_cnt = 0
        self.total_wafer_processed = 0
        self.no_action_count = 0
        self.episode_total_wait_time = 0
        self.episode_vtr_exe_time = 0
        self.episode_atr_exe_time = 0
        self.vtr_action_count = 0
        self.atr_action_count = 0
        logging.basicConfig(filename='6cb_2vtr_A2C_PPO_MPPO_DQN_2MSTEP_100WF.log', level=logging.INFO)
    
    # Helper method to get wafer state
    def _get_wafer_state(self, component):
        if component.holding_wafer is None:
            return 0
        elif component.holding_wafer.process_state == scd.WaferProcessState.BEFORE_PROCESS:
            return 0
        elif component.holding_wafer.process_state == scd.WaferProcessState.ALIGNED:
            return 1
        elif component.holding_wafer.process_state == scd.WaferProcessState.PROCESSED:
            return 2
        
    def get_observation(self):
        observation = []
        # Wafer presence (0 or 1)
        observation.append(1 if self.scd.components[scd.ComponentType.ENTRY].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.EXIT].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.VTR_LEFT_ARM].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.VTR_RIGHT_ARM].holding_wafer else 0)
        observation.extend([1 if chamber.holding_wafer else 0 for chamber in self.scd.components[scd.ComponentType.CHAMBER]])
        observation.append(1 if self.scd.components[scd.ComponentType.ATR_LEFT_ARM].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.ATR_RIGHT_ARM].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.ALIGNER].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.LP_IN].holding_wafer else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.LP_OUT].holding_wafer else 0)

        # Wafer process states (0: BEFORE_PROCESS, 1: ALIGNED, 2: PROCESSED)
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.ENTRY]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.EXIT]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.VTR_LEFT_ARM]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.VTR_RIGHT_ARM]))
        observation.extend([self._get_wafer_state(chamber) for chamber in self.scd.components[scd.ComponentType.CHAMBER]])
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.ATR_LEFT_ARM]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.ATR_RIGHT_ARM]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.ALIGNER]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.LP_IN]))
        observation.append(self._get_wafer_state(self.scd.components[scd.ComponentType.LP_OUT]))


        # Availability (0=N/A or 1)
        observation.append(1 if self.scd.components[scd.ComponentType.VTR_LEFT_ARM].process_time == 0 else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.VTR_RIGHT_ARM].process_time == 0 else 0)
        observation.extend([1 if chamber.process_time == 0 else 0 for chamber in self.scd.components[scd.ComponentType.CHAMBER]])
        observation.append(1 if self.scd.components[scd.ComponentType.ATR_LEFT_ARM].process_time == 0 else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.ATR_RIGHT_ARM].process_time == 0 else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.ALIGNER].process_time == 0 else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.LP_IN].process_time == 0 else 0)
        observation.append(1 if self.scd.components[scd.ComponentType.LP_OUT].process_time == 0 else 0)

        # vac_or_atm state for Entry and Exit
        observation.append(self.scd.components[scd.ComponentType.ENTRY].vac_or_atm)
        observation.append(self.scd.components[scd.ComponentType.EXIT].vac_or_atm)

        return np.array(observation, dtype=np.float32)

    def invalid_actions(self):
        invalid_vtr_actions = []
        invalid_atr_actions = []
        
        for vtr_action in scd.VTRAction:
            if not self.scd.check_vtr_action_validity(vtr_action):
                invalid_vtr_actions.append(vtr_action.value)
        
        for atr_action in scd.ATRAction:
            if not self.scd.check_atr_action_validity(atr_action):
                invalid_atr_actions.append(atr_action.value)
        
        return invalid_vtr_actions, invalid_atr_actions
    def get_min_exe_time_vtr_action(self, vtr_action_list):
        min_exe_time = float('inf')
        min_exe_time_action = None

        for action in vtr_action_list:
            vtr_action_enum = scd.VTRAction(action)
            vtr_component = self.scd.components[scd.ComponentType.VTR_LEFT_ARM] if "LEFT" in vtr_action_enum.name else self.scd.components[scd.ComponentType.VTR_RIGHT_ARM]
            
            # Determine target location for VTR action
            if "ENTRY" in vtr_action_enum.name:
                target_location = scd.VTRComponentLocation.ENTRY
            elif "EXIT" in vtr_action_enum.name:
                target_location = scd.VTRComponentLocation.EXIT
            elif "CHAMBER" in vtr_action_enum.name:
                chamber_parts = [part for part in vtr_action_enum.name.split('_') if part.isdigit()]
                if chamber_parts:
                    chamber_num = int(chamber_parts[0])
                    target_location = scd.VTRComponentLocation(chamber_num)
                else:
                    target_location = scd.VTRComponentLocation.ENTRY  # Default if no number found
            else:
                target_location = scd.VTRComponentLocation.ENTRY  # Default

            exe_time = self.scd.getVTRExeTime(vtr_component, target_location, update_location=False)
            
            if exe_time < min_exe_time:
                min_exe_time = exe_time
                min_exe_time_action = action

        return min_exe_time_action
    
    def get_min_exe_time_atr_action(self, atr_action_list):
        min_exe_time = float('inf')
        min_exe_time_action = None

        for action in atr_action_list:
            atr_action_enum = scd.ATRAction(action)
            atr_component = self.scd.components[scd.ComponentType.ATR_LEFT_ARM] if "LEFT" in atr_action_enum.name else self.scd.components[scd.ComponentType.ATR_RIGHT_ARM]
            
            # Determine target location for ATR action
            if "LP_TO" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.LP_IN
            elif "ALIGNER" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.ALIGNER
            elif "ENTRY" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.ENTRY
            elif "EXIT" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.EXIT
            elif "TO_LP" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.LP_OUT
            else:
                target_location = scd.ATRComponentLocation.LP_IN  # Default

            exe_time = self.scd.getATRExeTime(atr_component, target_location, update_location=False)
            
            if exe_time < min_exe_time:
                min_exe_time = exe_time
                min_exe_time_action = action

        return min_exe_time_action
    
    def step(self, action):
        vtr_action, atr_action = action
        
        # Get execution times before performing the action
        vtr_exe_time = 0
        atr_exe_time = 0

        if vtr_action != scd.VTRAction.NO_ACTION.value:
            vtr_action_enum = scd.VTRAction(vtr_action)
            vtr_component = self.scd.components[scd.ComponentType.VTR_LEFT_ARM] if "LEFT" in vtr_action_enum.name else self.scd.components[scd.ComponentType.VTR_RIGHT_ARM]
            
            # Determine target location for VTR action
            if "ENTRY" in vtr_action_enum.name:
                target_location = scd.VTRComponentLocation.ENTRY
            elif "EXIT" in vtr_action_enum.name:
                target_location = scd.VTRComponentLocation.EXIT
            elif "CHAMBER" in vtr_action_enum.name:
                # Safely extract chamber number
                chamber_parts = [part for part in vtr_action_enum.name.split('_') if part.isdigit()]
                if chamber_parts:
                    chamber_num = int(chamber_parts[0])
                    target_location = scd.VTRComponentLocation(chamber_num)
                else:
                    target_location = scd.VTRComponentLocation.ENTRY  # Default if no number found
            else:
                target_location = scd.VTRComponentLocation.ENTRY  # Default

            vtr_exe_time = self.scd.getVTRExeTime(vtr_component, target_location, update_location=False)

        if atr_action != scd.ATRAction.NO_ACTION.value:
            atr_action_enum = scd.ATRAction(atr_action)
            atr_component = self.scd.components[scd.ComponentType.ATR_LEFT_ARM] if "LEFT" in atr_action_enum.name else self.scd.components[scd.ComponentType.ATR_RIGHT_ARM]
            
            # Determine target location for ATR action
            if "LP_TO" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.LP_IN
            elif "ALIGNER" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.ALIGNER
            elif "ENTRY" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.ENTRY
            elif "EXIT" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.EXIT
            elif "TO_LP" in atr_action_enum.name:
                target_location = scd.ATRComponentLocation.LP_OUT
            else:
                target_location = scd.ATRComponentLocation.LP_IN  # Default

            atr_exe_time = self.scd.getATRExeTime(atr_component, target_location, update_location=False)


        # Perform action
        action_result, wafer_processed, busy_chamber_cnt, processed_wait_time = self.scd.perform_action(scd.VTRAction(vtr_action), scd.ATRAction(atr_action))


        # Update execution times and action counts
        if action_result > 0:
            if vtr_action != scd.VTRAction.NO_ACTION.value:
                self.episode_vtr_exe_time += vtr_exe_time
                self.vtr_action_count += 1
            if atr_action != scd.ATRAction.NO_ACTION.value:
                self.episode_atr_exe_time += atr_exe_time
                self.atr_action_count += 1

        # Determine reward and termination
        reward = 0
        terminated = False
        
        if action_result == 2:  # Both VTR and ATR actions were valid and performed
            reward = 0.2
        elif action_result == 1:  # Only one of VTR or ATR action was valid and performed
            reward = 0.1
        else:
            reward = -10
        
        self.episode_total_wait_time += processed_wait_time
        if wafer_processed:
            self.total_wafer_processed += 1
            reward = 1
            if self.total_wafer_processed == MAX_WAFER_NO:
                reward = 100
                terminated = True

        obs = self.get_observation()
        
        self.cumulative_reward += reward
        self.step_cnt += 1

        if self.step_cnt > MAX_STEP_LIMIT:
            terminated = True

        if self.render_mode == 'human':
            self.render()

        log_message = (
            f"__STEPCNT : {self.step_cnt} "
            f"__WF_PROCESSED : {self.total_wafer_processed} "
            f"__TOTAL_REWARDS : {self.cumulative_reward}"
        )
        info = {
            "log_message": log_message,
            "total_wafer_processed": self.total_wafer_processed,
            "avg_processed_wait_time": self.episode_total_wait_time / max(1, self.total_wafer_processed),
        }
        return obs, reward, terminated, False, info 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
            
        self.reset_cnt += 1

        # Calculate and log average execution times if it's not the first episode
        if self.reset_cnt > 1:
            avg_vtr_exe_time = self.episode_vtr_exe_time / max(1, self.vtr_action_count)
            avg_atr_exe_time = self.episode_atr_exe_time / max(1, self.atr_action_count)
            logging.info(f"Episode {self.reset_cnt - 1}: Avg VTR Exe Time: {avg_vtr_exe_time:.2f}, Avg ATR Exe Time: {avg_atr_exe_time:.2f}")

        self.cumulative_reward = 0
        self.step_cnt = 0
        self.total_wafer_processed = 0
        self.no_action_count = 0
        self.episode_total_wait_time = 0
        self.episode_vtr_exe_time = 0
        self.episode_atr_exe_time = 0
        self.vtr_action_count = 0
        self.atr_action_count = 0
        self.scd.reset(seed=seed)

        obs = self.get_observation()
        
        info = {}

        # Render environment
        if self.render_mode == 'human':
            self.render()

        # Return observation and info
        return obs, info

    def render(self):
        self.scd.render(self.cumulative_reward)

# For unit testing
if __name__=="__main__":

    env = gym.make('eqp-scheduler-v0', render_mode='human')

    # Use this to check our custom environment
    print("Check environment begin")
    check_env(env.unwrapped)
    print("Check environment end")

    # Reset environment
    obs = env.reset()[0]
    # 수정된 부분: action_space가 Tuple인 경우를 처리
    if isinstance(env.action_space, gym.spaces.Tuple):
        num_vtr_actions = env.action_space.spaces[0].n
        num_atr_actions = env.action_space.spaces[1].n
        print(f"VTR actions: {num_vtr_actions}, ATR actions: {num_atr_actions}")
    else:
        raise ValueError("Unexpected action space structure")

    # Take some random actions
    for i in range(10000):
        invalid_vtr_actions, invalid_atr_actions = env.invalid_actions()
        vtr_action = random.choice([a for a in range(num_vtr_actions) if a not in invalid_vtr_actions])
        atr_action = random.choice([a for a in range(num_atr_actions) if a not in invalid_atr_actions])
        obs, reward, terminated, _, _ = env.step((vtr_action, atr_action))
        if terminated:
            obs = env.reset()[0]
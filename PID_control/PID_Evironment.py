from Water_evironment import Water_level_in_tank as water_tank
from All_Agent import PID_Agent
import matplotlib.pyplot as plt
import numpy as np


class PID_control_watel:
    def __init__(self,radius=5, max_height=10, setpoint=5, Cv_flow_in = 2.8,Cv_flow_out = 0.5,
                 characteristics_valve = 'linear',dt = 0.1,spect=dict()):
        self.setpoint = setpoint
        self.dt = dt
        self.characteristics_valve = characteristics_valve
        self.PID = PID_Agent()
        self.Water_tank = water_tank(radius=radius,max_height=max_height,setpoint=setpoint,Cv_flow_in=Cv_flow_in,Cv_flow_out=Cv_flow_out,devitive=0,tolerance=0.1,characteristics_valve=self.characteristics_valve,dt=self.dt)
        self.integral = 0
        self.prev_error = 0
        self.water_level = 0

        self.water_level_history = []
        self.info = dict()
        self.reward = 0

        self.offset_update = float(1e7)
        self.settling_update =  float(1e7)
        self.time_diff_update =  float(1e7)

        self.spect_overshoot = spect.get('Percent_Overshoot',20)
        self.spect_undershoot =spect.get('Percent_Undershoot',0)

    def reset(self):
        self.water_level = self.Water_tank.reset()
        self.integral = 0
        self.reward = 0
        self.prev_error = 0
        self.water_level_history = []
        self.info = {}
        return self.water_level,self.reward,list(self._get_state())
        

    def step(self, Kp, Ki, Kd):
        error = self.setpoint - self.water_level
        Action = self.PID.compute(error=error, Kp=Kp, Ki=Ki, Kd=Kd,dt=self.dt)
        state, _, done_evn, _, _ = self.Water_tank.step(Action_Agent=Action)

        self.water_level = state
        self.water_level_history.append(self.water_level)
    
        if (len(self.water_level_history) > 2 and (len(self.water_level_history) % 1000 == 0)):
            self.info = self.stepinfo(self.water_level_history,self.setpoint)
            self.reward = self.reward_Control(self.info)

  
        
        

        return self.water_level,self.reward,list(self._get_state())
    
    def _get_state(self):
        
        start_fall,end_fall,start_rise,end_rise  = self.info.get('Start_Fall',None),self.info.get('End_Fall',None),self.info.get('Start_Rise',None),self.info.get('End_Rise',None)
        percent_overshoot,percent_undershoot,settling_time = self.info.get('Percent_Overshoot',0),self.info.get('Percent_Undershoot',0),self.info.get('Settling_Time',None)
        offset = self.info.get('Offset',0)
        time_diff = 0

        if isinstance(start_fall,(np.integer,np.floating)) and isinstance(end_fall,(np.integer,np.floating)):
            time_diff = end_fall - start_fall
        elif isinstance(start_rise,(np.integer,np.floating)) and isinstance(end_rise,(np.integer,np.floating)):
            time_diff = end_rise - start_rise
        else:
            time_diff = len(self.water_level_history)
        
        if settling_time is None:
            settling_time = len(self.water_level_history)

        return time_diff,percent_overshoot,percent_undershoot,settling_time,offset
              
    def stepinfo(self,history,setpoint,tolerance = 0.02,consecutive = 0.3):
        
        count = len(history)
        consecutive_step = round(count * consecutive)

        #find the steady state
        referRence_yss = None
        settling_time = None
        y_ss = np.mean(history[-(consecutive_step):])

        tolerance_band = np.abs(y_ss) * tolerance
        upper_bound = (y_ss + tolerance_band)
        lower_bound = (y_ss - tolerance_band)
        count_ss = 0
        for i,history_list in enumerate(history):
            if lower_bound <= history_list <= upper_bound:
                count_ss +=1
            else:
                count_ss = 0
            if count_ss >= (consecutive_step):
                    settling_time = i - (consecutive_step - 1)
                    break
            
    
        if isinstance(settling_time,int):
            referRence_yss = np.mean(history[settling_time:])
            
            
        else:
            referRence_yss = y_ss
        
        
        #find fall time  of process
        upper_fall = float('inf')
        start_fall,end_fall = None,None
        fall_10,fall_90 = 0,0
        count_fall = 0

        for i in range(1,count -1):
            if history[0] <= (referRence_yss + tolerance_band):
                break

            if history[i] < upper_fall:
                upper_fall = history[i]
                count_fall += 1
                end_fall = i

            if history[i] > history[i - 1] and history[i] > upper_fall * 1.001 :
                end_fall = i
                break

            if i > round(consecutive_step * 0.1) and count_fall >= round(consecutive_step * 0.1) and start_fall is None:
                start_fall = round(i - (consecutive_step * 0.1))
        
        if end_fall is not None and start_fall is not None:
            dy_fall = (history[start_fall]) - history[end_fall]
            y_10_fall = 0.1 * dy_fall
            y_90_fall = 0.9 * dy_fall

            dy_10_fall = (history[start_fall] - y_10_fall)
            dy_90_fall = (history[start_fall] - y_90_fall)
            

        
            fall_10 = np.argmin(np.abs(np.array(history[start_fall:end_fall]) - dy_10_fall))
            fall_90 = np.argmin(np.abs(np.array(history[start_fall:end_fall]) - dy_90_fall))
        else:
            pass
        
        
        if fall_10 == fall_90:
            fall_10,fall_90 = None,None

        #find rise time of process
        end_rise = None
        rise_10,rise_90 = None,None

        if (start_fall  is  None) and (end_fall is None):
            for i in range(0,count - 1):
                if history[i] >= referRence_yss:
                    end_rise = i
                    break
        
            dy_rise = referRence_yss - history[0]
            dy_10_rise = 0.1 * dy_rise
            dy_90_rise = 0.9 * dy_rise

            y_10_rise = history[0] + dy_10_rise
            y_90_rise = history[0] + dy_90_rise
            if end_rise > 0:
                rise_10 = np.argmin(np.abs(np.array(history[0:end_rise]) - y_10_rise))
                rise_90 = np.argmin(np.abs(np.array(history[0:end_rise]) - y_90_rise))

                
        
        #find overshoot

        tolerance_band = np.abs(referRence_yss) * tolerance
        upper_bound = (referRence_yss + tolerance_band)
        lower_bound = (referRence_yss - tolerance_band)
    
        where_overshoot = 0
        where_undershoot = 0
        percent_overshoot = 0
        percent_undershoot = 0
        undershoot_list = []
        overshoot_list = []
        start_count_shoot  = 0

        if end_fall is not None and end_fall != 1:
            start_count_shoot = end_fall
        elif end_rise is not None:
            start_count_shoot = end_rise
        else:
            start_count_shoot = 0


        for i in range(start_count_shoot,count -1):
            if history[i] > (upper_bound):
                overshoot_list.append(history[i])
            else:
                overshoot_list.append(float('-inf'))
            
            if history[i] < (lower_bound):
                undershoot_list.append(history[i])
            else:
                undershoot_list.append(float('inf'))

        where_overshoot = np.argmax(overshoot_list)
        where_undershoot = np.argmin(undershoot_list)
        where_overshoot += start_count_shoot
        where_undershoot += start_count_shoot

        if (where_overshoot == 0) or  (history[where_overshoot] < upper_bound) :
            where_overshoot = None

        if (where_undershoot == 0) or (history[where_undershoot] > lower_bound):
            where_undershoot = None

        if where_overshoot is not None:
            percent_overshoot = np.abs(history[where_overshoot] - referRence_yss)/np.abs(referRence_yss) *100

        if where_undershoot is not None:
            percent_undershoot = np.abs(history[where_undershoot] - referRence_yss)/np.abs(referRence_yss) *100

        #find offset
        offset = setpoint - referRence_yss

    
        info = {'Start_Fall':fall_10,
                'End_Fall':fall_90,
                'Start_Rise':rise_10,
                'End_Rise':rise_90,
                'Where_Max_Overshoot':where_overshoot,
                'Where_Max_Undershoot':where_undershoot,
                'Percent_Overshoot':percent_overshoot,
                'Percent_Undershoot':percent_undershoot,
                'Settling_Time':settling_time,
                'Steady_State':referRence_yss,
                'Upper_Tolerance':upper_bound,
                'Lower_Tolerance':lower_bound,
                'Offset':offset,
                'Size_data':count,
                }


        return info

    def reward_Control(self,info : dict):
        
        start_fall,end_fall,start_rise,end_rise  = info.get('Start_Fall',None),info.get('End_Fall',None),info.get('Start_Rise',None),info.get('End_Rise',None)
        percent_overshoot,percent_undershoot,settling_time = info.get('Percent_Overshoot',None),info.get('Percent_Undershoot',None),info.get('Settling_Time',None)
        size_data = info.get('Size_data',None)
        offset = info.get('Offset',0)

        #parameter adjust
        K1,K2,K3 = 0.1,0.8,0.1
        bese_range = size_data * 0.1
        time_start = 0
        All_reward = 0
        
        
        if isinstance(start_fall,(np.integer,np.floating)) and isinstance(end_fall,(np.integer,np.floating)):
            time_start = end_fall - start_fall
        elif isinstance(start_rise,(np.integer,np.floating)) and isinstance(end_rise,(np.integer,np.floating)):
            time_start = end_rise - start_rise
        else:
            return 0
        #print(time_start)
        #parameter adjust
        K1,K2,K3,K4 = 0.3,0.8,0.1,0.5
        bese_range = size_data * 0.1

        # reward_fast = np.exp( -K1 * (time_start / bese_range))
        # reward_over = np.exp( - K2 * (percent_overshoot/100))
        # reward_under = np.exp( -K3 * (percent_undershoot/100))

        reward_fast = - time_start
        reward_over = - percent_overshoot
        reward_under = - percent_undershoot
        reward_offset = - np.abs(offset)

        #reward_over = reward_over if percent_overshoot <= self.spect_overshoot else reward_over * -1 * 10
        #reward_under = reward_under if percent_undershoot <= self.spect_undershoot else reward_under * -1 *10

        All_reward = reward_fast + reward_over + reward_under + reward_offset
        
        if settling_time is None:
            reweard_settling = (All_reward**2)
        else:
            #reweard_settling = np.exp( -K4 * (settling_time / bese_range))
            reweard_settling = - settling_time
        
        return All_reward + reweard_settling

    def check_done(self):
        start_fall,end_fall,start_rise,end_rise  = self.info.get('Start_Fall',0),self.info.get('End_Fall',self.time_diff_update),self.info.get('Start_Rise',0),self.info.get('End_Rise',self.time_diff_update)
        percent_overshoot,percent_undershoot,settling_time = self.info.get('Percent_Overshoot',0),self.info.get('Percent_Undershoot',0),self.info.get('Settling_Time',float('inf'))
        offset = self.info.get('Offset',0)
        time_diff = 0
        Done_status_time_diff,Done_status_settling,Done_status_offset,Done_status_overshoot,Done_status_undershoot = False,False,False,False,False

        if isinstance(start_fall,(np.integer,np.floating)) and isinstance(end_fall,(np.integer,np.floating)):
            time_diff = end_fall - start_fall
        elif isinstance(start_rise,(np.integer,np.floating)) and isinstance(end_rise,(np.integer,np.floating)):
            time_diff = end_rise - start_rise
        else:
            time_diff = len(self.water_level_history)

        settling_time_cus = len(self.water_level_history) if settling_time is None else settling_time
        
        if settling_time_cus <= len(self.water_level_history):
            Done_status_time_diff = time_diff <= self.time_diff_update + (self.time_diff_update * 0.5)
            self.time_diff_update  = time_diff if time_diff < self.time_diff_update else self.time_diff_update
        
        Done_status_offset = True if np.abs(offset) < (0.02 * self.setpoint) else False

        Done_status_settling  = settling_time_cus <= self.settling_update + (self.settling_update * 0.5) and settling_time_cus is not None
        self.settling_update = settling_time_cus if settling_time_cus < self.settling_update else self.settling_update

        Done_status_overshoot = percent_overshoot <= self.spect_overshoot
        #self.spect_overshoot = percent_overshoot if percent_overshoot < self.spect_overshoot else self.spect_overshoot

        Done_status_undershoot = percent_undershoot <= self.spect_undershoot
        #self.spect_undershoot = percent_undershoot if percent_undershoot < self.spect_undershoot else self.spect_undershoot

       

        Done = (
            Done_status_time_diff == Done_status_settling ==
            Done_status_overshoot == Done_status_undershoot ==
            Done_status_offset
        )

        print(
            f'time diff: {self.color_bool(Done_status_time_diff)} '
            f'/update: {self.time_diff_update} | current time diff: {time_diff} | '
            f'settling: {self.color_bool(Done_status_settling)} /update: {self.settling_update} '
            f'| state: {settling_time} | over: {percent_overshoot} status: {self.color_bool(Done_status_overshoot)} '
            f'\nunder: {percent_undershoot} status: {self.color_bool(Done_status_undershoot)} '
            f'| offset: {self.color_bool(Done_status_offset)} '
            f'| -> Done: {self.color_bool(Done)}\n'
            f'\n--------------------------------------------------------'
        )
        return Done
    
    def color_bool(self,val: bool) -> str:
        return f"\033[32m{val}\033[0m" if val else f"\033[31m{val}\033[0m"





        

            
        



        



    



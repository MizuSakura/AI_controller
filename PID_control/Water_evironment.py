import numpy as np

class controlvalve_signal:
    def __init__(self, tau = 1.0 ,delay_step = 3, dt= 0.1,Typevalve ='linear'):
        self.tau = tau #Time constant บอกว่า system ตอบสนอง เร็วหรือช้า
        self.delay_step = delay_step #Delay steps จำนวน step ที่ output เริ่มเปลี่ยนแปลงช้า (dead time)
        self.dt = dt #Sampling time เวลาระหว่าง การอ่านค่าและอัปเดต output 
        self.control_buffer = [0.0] * self.delay_step
        self.output_signal = 0.0

        self.Kv = 1.652 # Cv flow rate gain  m^3/h
        self.detal_Pressure = 1 #Bar
        self.density = 1000
        self.Rangeability  = 5
        self.exponent  = 0.3
        self.Typevalve = Typevalve
   
    def reset(self):
        self.output_signal = 0.0
        self.control_buffer = [0.0] * self.delay_step
        return self.output_signal
    
    def step_signal(self,signal_command):
        output_comand = self.control_buffer.pop(0)
        self.control_buffer.append(signal_command)
        dy = (output_comand - self.output_signal) * self.dt / self.tau
        self.output_signal += dy
        return self.output_signal
    
    def Characteristics_valve(self,signal_command = 0):
        if self.Typevalve == 'linear':
            Flow_rate = self.Kv * np.sqrt(self.detal_Pressure) * self.step_signal(signal_command)
        elif self.Typevalve == 'Equal_Percentage':
            Flow_rate = self.Kv * np.exp(self.Rangeability * (self.step_signal(signal_command) - 1))
        elif self.Typevalve == 'Quick_Opening':
            Flow_rate = self.Kv * (self.step_signal(signal_command) ** self.exponent)

        return Flow_rate
    
class Water_level_in_tank:
    def __init__(self,radius=5, max_height=10, setpoint=5, Cv_flow_in = 2.8,
                Cv_flow_out = 0.5,devitive = 0.1,tolerance = 0.1,characteristics_valve = 'linear',dt = 0.1):
        self.radius = radius
        self.max_height = max_height
        self.area = np.pi * radius **2 #m^2
        self.setpoint = setpoint
        self.Cv_flow_in = Cv_flow_in #GPM/s
        self.Cv_flow_out = Cv_flow_out #GPM/s

        self.area_outlet_hold = 0.2
        self.gravity = 9.81 #m/s^2
        self.Flow_out = 0
        self.Flow_in = 0

        self.devitive = devitive
        self.tolerance = tolerance
        self.Previous_Error = 0
        self.Previous_Action = 0

        self.tau = 1
        self.delay_step = 3
        self.dt = dt
        self.characteristics_valve = characteristics_valve
        self.Controlvalve = controlvalve_signal(tau=self.tau, delay_step= self.delay_step, Typevalve= self.characteristics_valve)
        self.t = 0

    def reset(self):
        self.current_level = np.random.uniform(0,(self.max_height * self.devitive))
        self.prev_error = self.setpoint - self.current_level
        return self.get_state()
    
    def get_state(self):
        return float(self.current_level)
    
    def step(self,Action_Agent,Add_noise = False):
        Action = np.clip(Action_Agent, 0 ,1)
        self.Flow_in = self.Controlvalve.Characteristics_valve(Action)
        self.Flow_out = self.Cv_flow_out * self.area_outlet_hold * np.sqrt( 2 * self.gravity * self.current_level)
        

        devitive_level = (self.Flow_in - self.Flow_out) / self.area
        self.current_level += devitive_level * self.dt
        self.current_level = np.clip(self.current_level, 0 ,self.max_height)

        Error = self.setpoint - self.current_level
        Done = abs(Error) <= self.tolerance

        reward = self.Reward_funtion(Error)

        self.Previous_Error = Error
        self.Previous_Action = Action
        self.t += 1


        return self.get_state(), float(reward), Done , float(Error),float(self.Previous_Error)
    

    def Reward_funtion(self,Current_error):
        parameter_Adjust = self.max_height/ (0.3 * self.max_height ** 2)
        reward = np.exp( -(parameter_Adjust) * (np.abs(Current_error)))
        return reward
    
    def sin_noise_random(self,Flow,t, A=10, f=1, phi_range=1):
        random_phi = np.random.uniform(-phi_range, phi_range, size=np.shape(t))
        return Flow * np.abs((A * np.sin((2 * (np.pi/4) * f * t) + random_phi)))
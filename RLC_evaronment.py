import numpy as np

class RC_environment:
    def __init__(self, R=1.0, C=1.0, dt=0.01,setpoint = 5):
        self.R = R  # Resistance in ohms
        self.C = C  # Capacitance in farads
        self.dt = dt  # Time step in seconds
        self.voltage_capacitor = 0.0  # Initial voltage across the capacitor
        

        self.setpoint = setpoint
        self.maximumn_volt = 10

        self.time = 0.0
        self.round_reset = 0
        self.per_error = 0
        self.per_action = 0
        self.penalize_count = 0
        self.intergal_error = 0


        self.reset()

    def reset(self):
        if self.round_reset < 100:
            self.voltage_capacitor = np.random.uniform(0,max(0,(self.round_reset/100)*self.maximumn_volt))
        else:
            self.voltage_capacitor = np.random.uniform(0, self.maximumn_volt)
        
        self.per_error = self.setpoint - self.voltage_capacitor
        self.round_reset +=1
        self.penalize_count = 0
        self.time = 0.0
        reward = 0
        Done = False
        self.intergal_error = 0

        return self.voltage_capacitor,reward,Done,self.per_error,self.per_action,self.intergal_error,0
    
    def step(self, voltage_source):
        deltal_volt = (voltage_source - self.voltage_capacitor) / (self.R * self.C)
        self.voltage_capacitor += deltal_volt * self.dt
        self.time += self.dt

        error = self.setpoint - self.voltage_capacitor
        reward = self.reward_function(abs(error))

        Done = abs(error) <= 0.1

        self.per_error = error
        self.per_action = voltage_source
        self.intergal_error += error * self.dt

        return float(self.voltage_capacitor) , float(reward) ,Done,self.per_error,self.per_action,self.intergal_error,deltal_volt
    
    def reward_function(self,error):

        parameter = self.setpoint / (0.3 * self.setpoint**2)
        reward = np.exp(-(parameter)*(error))
        reward_delta = self.reward_delta(error=error,per_error=self.per_error)

        return reward+(reward_delta)

    def reward_delta(self,error,per_error):
        delta_error = abs(per_error) - abs(error)
        waight = 101
        if delta_error > 0:
           
            self.penalize_count = 0
            delta_error = delta_error * waight
        else:
            self.penalize_count +=0.1
            delta_error = -abs(delta_error* waight * (1 + self.penalize_count))
        return delta_error
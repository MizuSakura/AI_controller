import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
import time 
from datetime import datetime
import os
from PIL import Image, ImageTk
from PID_Evironment import PID_control_watel
import torch
from All_Agent import DDPG_Agent
from Water_evironment import Water_level_in_tank

class PIDSimulatorGUI:
    def __init__(self, MW):
        self.MW = MW
        MW.title("PID Simulator with 666 Agent")
        MW.geometry("1280x720")
        MW.configure(bg="#2310D1")

        MW.grid_rowconfigure(0, weight=0)
        MW.grid_rowconfigure(1, weight=0)

        MW.grid_columnconfigure(0, weight=0, minsize=350)
        MW.grid_columnconfigure(1, weight=1, minsize=20)
        MW.grid_columnconfigure(2, weight=0, minsize=400)
    #############################

    ################### Water Tank Parameters ########################
        self.radius_var = tk.DoubleVar(value=5.0)
        self.max_height_var = tk.DoubleVar(value=10.0)
        self.setpoint_var = tk.DoubleVar(value=5.0)
        self.cv_flow_in_var = tk.DoubleVar(value=2.8)
        self.cv_flow_out_var = tk.DoubleVar(value=0.5)
        self.devitive_var = tk.DoubleVar(value=0.1)
        self.tolerance_var = tk.DoubleVar(value=0.1)
        self.characteristics_valve_var = tk.StringVar(value='linear') # เป็น String
        self.dt_var = tk.DoubleVar(value=0.1)
        self.max_minutes_var = tk.DoubleVar(value=60.0)
        self.Episode_var = tk.IntVar(value=1)
    ############################### PID  ##############################################
       

        self.agent = DDPG_Agent(State_dimension=9)
        #self.pid = PID_control_watel(characteristics_valve='Equal_Percentage')

        self.kp = tk.DoubleVar(value=1.0)
        self.ki = tk.DoubleVar(value=0.002)
        self.kd = tk.DoubleVar(value=0.0)
 
        self.setup_gui()
        self.setup_gui_env()
        self.setup_gui_status()

#################################################################################################


    def setup_gui_env(self):

        self.SETENV = tk.LabelFrame(self.MW, text="Setup Environment", padx=5, pady=5)
        self.SETENV.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.SETENV.configure(bg="#FEFEFF") 
        row_idx = 0

        for i in range(self.SETENV.grid_size()[1]):
            self.SETENV.grid_rowconfigure(i, weight=1)
        self.SETENV.grid_columnconfigure(0, weight=0) # Label column
        self.SETENV.grid_columnconfigure(1, weight=1) # Entry column
        


        #SETENV_section = tk.Label(SETENV, text="Tank Parameters", font=("Arial", 12, "underline"))
        #SETENV_section.grid(row=4, column=0, sticky="w", padx=5, pady=5)
        #SETENV_section_separator = ttk.Separator(SETENV, orient='horizontal')
        #SETENV_section_separator.grid(row=4, column=0, columnspan=2, sticky="ew", pady=15)


         ######################################################## Water Tank Parameters #####################################

        tk.Label(self.SETENV, text="Radius:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.radius_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1
        
        tk.Label(self.SETENV, text="Max Height:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.max_height_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1
        
        tk.Label(self.SETENV, text="Setpoint:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.setpoint_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # Cv_flow_in
        tk.Label(self.SETENV, text="Cv Flow In:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.cv_flow_in_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1
        
        # Cv_flow_out
        tk.Label(self.SETENV, text="Cv Flow Out:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.cv_flow_out_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # Devitive (Deviate?) - Assuming it's 'Derivative' or 'Deviation'
        tk.Label(self.SETENV, text="Derivative:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.devitive_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # Tolerance
        tk.Label(self.SETENV, text="Tolerance:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.tolerance_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # Characteristics Valve (Consider using OptionMenu for predefined values later)
        tk.Label(self.SETENV, text="Valve Type:", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.characteristics_valve_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        # dt
        tk.Label(self.SETENV, text="Delta Time (dt):", font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.dt_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1
        ################################################################################################################################


        ################################### PID Parameters ##########################################################
        tk.Label(self.SETENV, text="Kp:",font=("Arial", 10)).grid(row=row_idx, column=0,sticky="w",padx=5, pady=5)
        Kp_entry = tk.Entry(self.SETENV, textvariable=self.kp,font=("Arial", 10))
        Kp_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1
        

        tk.Label(self.SETENV, text="Ki:",font=("Arial", 10)).grid(row=row_idx, column=0,sticky="w",padx=5, pady=5)
        Ki_entry = tk.Entry(self.SETENV, textvariable=self.ki,font=("Arial", 10))
        Ki_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1
    
        tk.Label(self.SETENV, text="Kd:",font=("Arial", 10)).grid(row=row_idx, column=0,sticky="w",padx=5, pady=5)
        Kd_entry = tk.Entry(self.SETENV, textvariable=self.kd,font=("Arial", 10))
        Kd_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        
        self.Episode_var = tk.IntVar(value=1)
        tk.Label(self.SETENV, text = "Episode:",font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w")
        self.Episode_entry = tk.Entry(self.SETENV, font=("Arial", 10), textvariable=self.Episode_var)
        self.Episode_entry.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

        tk.Label(self.SETENV, text="Max minutes:",font=("Arial", 10)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.SETENV, textvariable=self.max_minutes_var, font=("Arial", 10)).grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
        row_idx += 1

         ###########################################################################


    def setup_gui_status(self):

        self.RF = tk.LabelFrame(self.MW, text="Monitor", padx=5, pady=5) 
        self.RF.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        self.RF.configure(bg="#F6F6F6")

        self.RF.grid_rowconfigure(0, weight=1)
        self.RF.grid_columnconfigure(0, weight=1)
        #### status label
        self.Status = tk.LabelFrame(self.MW, text="Training Status", padx=10, pady=10)
        self.Status.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=(5, 10))

        self.Status.grid_rowconfigure(0, weight=1) 
        self.Status.grid_columnconfigure(0, weight=1)
        ########
       
        ###### widgets in Status
        self.elapsed_time_label = tk.Label(self.Status, text="Elapsed Time: 00:00:00", font=("Arial", 10))
        self.elapsed_time_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        self.status_message_label = tk.Label(self.Status, text="Status: Idle", font=("Arial", 10))
        self.status_message_label.grid(row=1, column=0, sticky="w", padx=5, pady=5)

        self.Status.grid_rowconfigure(0, weight=0) # Elapsed Time ไม่ต้องขยาย
        self.Status.grid_rowconfigure(1, weight=0) # Status Message ไม่ต้องขยาย

        #########
        
    def setup_gui(self):
       

        tk.Button(self.MW, text="Run Simulation", command=self.run_simulation).grid(row=2, column=0, columnspan=3, sticky="s", pady=(5, 10))

        

    def run_simulation(self):

        radius = self.radius_var.get()
        max_height = self.max_height_var.get()
        setpoint = self.setpoint_var.get() # setpoint สำหรับถังน้ำ
        cv_flow_in = self.cv_flow_in_var.get()
        cv_flow_out = self.cv_flow_out_var.get()
        devitive = self.devitive_var.get() # ถ้า devitive, tolerance เป็นคุณสมบัติของ Tank
        tolerance = self.tolerance_var.get() # หรือเป็นค่าที่ส่งให้ PID Controller แต่มาจาก GUI
        characteristics_valve = self.characteristics_valve_var.get()
        dt = self.dt_var.get()
        
        self.pid = PID_control_watel(
            radius=radius,
            max_height=max_height,
            setpoint=setpoint,
            Cv_flow_in=cv_flow_in,
            Cv_flow_out=cv_flow_out,
            characteristics_valve=characteristics_valve,
            dt=dt,
            devitive=devitive,   # 📌 ส่งค่า devitive จาก GUI ไปยัง PID_control_watel
            tolerance=tolerance  # 📌 ส่งค่า tolerance จาก GUI ไปยัง PID_control_watel
            # spect=... ถ้าท่านมี GUI input สำหรับ spect ด้วย
        )
        
        #self.pid.Water_tank = Water_level_in_tank(radius=radius, max_height=max_height, setpoint=setpoint, cv_flow_in=cv_flow_in, cv_flow_out=cv_flow_out, devitive=devitive, tolerance=tolerance, characteristics_valve=characteristics_valve, dt=dt)
        Episode = self.Episode_var.get()
        kp_initial = self.kp.get() # ค่า Kp, Ki, Kd เริ่มต้นจาก GUI
        ki_initial = self.ki.get()
        kd_initial = self.kd.get()
        max_minutes = self.max_minutes_var.get()

        Episode = self.Episode_var.get()
        kp, ki, kd = self.kp.get(), self.ki.get(), self.kd.get()
        agent = self.agent
        pid = self.pid

        for j in range(Episode):
            reward_list = []
            water_level, reward, state = pid.reset()
            state = [kd, ki, kp, water_level] + state
            state_input = torch.tensor([state], dtype=torch.float32).to(agent.device)

            action = agent.Select_Action(state_input, True)
            kp, ki, kd = action[0], action[1], action[2]

            water_history, t = [], []
            for i in range(5000):
                water_level, reward, state_output = pid.step(kp, ki, kd)
                state_output = [kd, ki, kp, water_level] + state_output
                water_history.append(water_level)
                t.append(i)
                reward_list.append(reward)

            done = pid.check_done()
            next_state = torch.tensor([state_output], dtype=torch.float32).to(agent.device)
            agent.Replay_memory.Store(State=state_input, Action=action, Reward=reward, Next_State=next_state, Done=done)
            agent.Optimize_model()

            print(f"Episode {j+1}/{Episode} - Kp: {kp:.4f}, Ki: {ki:.4f}, Kd: {kd:.4f}, Reward: {reward:.4f}")

        info = pid.stepinfo(water_history, pid.setpoint)
        self.plot_result(t, water_history, info, reward)



    def plot_result(self, t, water_history, info, reward):
        #for widget in self.canvas_frame.winfo_children():
            #widget.destroy()

        plt.figure(figsize=(8, 4))
        plt.plot(t, water_history, label="Water Level")
        plt.axhline(self.pid.Water_tank.setpoint, linestyle='--', color="#ff0000", label='Setpoint')


        color_list = {
            'Start_Fall': "#00ff04", 'End_Fall': "#ff0000",
            'Start_Rise': "#2fff00", 'End_Rise': "#ff0000",
            'Where_Max_Overshoot': "#ff0000", 'Where_Max_Undershoot': "#0084ff",
            'Settling_Time': "#00fbff", 'Steady_State': "#00ffee",
            'Upper_Tolerance': "#fff200", 'Lower_Tolerance': "#00ff11"
        }

        for key in ['Start_Fall','End_Fall','Start_Rise','End_Rise','Where_Max_Overshoot','Where_Max_Undershoot','Settling_Time']:
            if info.get(key) is not None:
                plt.axvline(info[key], label=key, linestyle='-.', color='#00ff00', alpha=0.4)
        for key in ['Steady_State', 'Upper_Tolerance', 'Lower_Tolerance']:
            if info.get(key) is not None:
                plt.axhline(info[key], label=key, linestyle='-.', color='#ff9900', alpha=0.4)


        plt.title(f"Reward: {reward:.4f}")
        plt.grid()
        plt.legend()
        plt.show() 

        #canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        #canvas.draw()
        #canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == '__main__':
    root = tk.Tk()
    app = PIDSimulatorGUI(root)
    root.mainloop()

import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import time 
from WaterTank import WaterTankEnv
from DDPG_Agent import DDPG_Agent
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import time
from datetime import datetime
from matplotlib.figure import Figure
import os
import pandas as pd
import sys
from tkinter import messagebox
import threading



# Agent & Env 
env = WaterTankEnv(setpoint= 5)
Agent = DDPG_Agent(
    max_level = 10,
    learning_rate=0.001, 
    replay_buffer_size =10000,
    batch_size=256
)
##################################### MAIN WINDOW ##########################################################
MW= tk.Tk()
MW.title("Tank Simulator")
MW.geometry("1600x900")
MW.configure(bg="#08086A")

SETENV = tk.LabelFrame(MW, text="Setup Environment", padx=5, pady=5)
SETENV.place(x=40, y=80, width=600, height=620)
SETENV.configure(bg="#FFFFFF") 

SETENV_section = tk.Label(SETENV, text="Tank Parameters", font=("Arial", 12, "underline"))
SETENV_section.grid(row=4, column=0, sticky="w", padx=5, pady=5)
SETENV_section_separator = ttk.Separator(SETENV, orient='horizontal')
SETENV_section_separator.grid(row=4, column=0, columnspan=2, sticky="ew", pady=15)

RF = tk.LabelFrame(MW, text="Monitor", padx=5, pady=5)
RF.place(x=660, y=80, width=600, height=620)
RF.configure(bg="#FFFFFF")

Status = tk.LabelFrame(MW, text="Training Status", padx=10, pady=10)
Status.place(x=660, y=700-80, width=600, height=120)
################################################################ MW ##########################################################


     
    
###################################### SETUP ENVIRONMENT ##########################################################
env = WaterTankEnv(setpoint= 5)
 
########################## SAVEE PATH
def create_save_path(Save_dir, episode=None, filename="plot.png"):
    date_str = datetime.now().strftime("%d_%m_%Y")
    folder_path = os.path.join(Save_dir, date_str)
    if episode is not None:
        folder_path = os.path.join(folder_path, f"episode_{int(episode):03d}")
    os.makedirs(folder_path, exist_ok=True)
    full_path = os.path.join(folder_path, filename)
    return full_path

def Browse_save_path():
    save_path = filedialog.askdirectory()
    if save_path:
        Save_entry.delete(0, tk.END)
        Save_entry.insert(0, save_path)

def on_create_path():
    Save_dir = Save_entry.get()
    episode = Episode_entry.get()
    episode = episode if episode.strip() else None

    try:
        path = create_save_path(Save_dir, episode)
        messagebox.showinfo("Success", f"Path created:\n{path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create path:\n{e}")

                                            # SAVE PATH ################################################

 ######################################## THREADING ##########################################################                                       
def threaded_training():
    threading.Thread(target=start_training).start()
##########################################################


 ############### Agent 47 ####################################  DEF
def test_agent(env, Agent, run_time=20, episode=0):
    state = env.reset()
    levels, valves, errors, rewards, timestamps = [], [], [], [], []
    start_time = time.time()

    while time.time() - start_time < run_time:
        action = Agent.select_action(state, False)
        next_state, reward, done, error, _ = env.step(action)
        state = next_state

        levels.append(state)
        valves.append(action)
        errors.append(error)
        rewards.append(reward)
        timestamps.append(time.time() - start_time)
        time.sleep(0.001)

        
    fig = Figure(figsize=(6, 4), dpi=100)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(timestamps, levels, label="Water Level (m)")
    ax1.axhline(env.setpoint, color='r', linestyle='--', label="Target Level")
    ax1.set_ylabel("Level (m)")
    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(timestamps, valves, label="Valve Signal", color='g')
    ax2.set_ylabel("Action (0-1)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid()
    save_path = create_save_path(Save_entry.get(), filename=f'Evaluation_Phase_{episode}.png')
    fig.savefig(save_path)
    fig.suptitle("Performance of Agent SNAKE")
    return fig


                                # AGEnT 47############################################


##### Save plot  ##########################################################
def saveplot(plot_reward, plot_loss_Actor, plot_loss_Critic):
    
    x = list(range(1, 1 + len(plot_reward)))
    fig2 = Figure(figsize=(6, 4), dpi=100)
    date_str = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")

    ax1 = fig2.add_subplot(2, 1, 1)
    ax1.plot(x, plot_reward, label="Reward")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward per Episode")
    ax1.grid()
    ax1.legend()

    ax2 = fig2.add_subplot(2, 1, 2)
    ax2.plot(plot_loss_Actor, label="Loss Actor", color='r')
    ax2.plot(plot_loss_Critic, label="Loss Critic", color='b')
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Time step")
    ax2.set_title("Loss over Time")
    ax2.grid()
    ax2.legend()

    fig2.tight_layout()
    fig2.subplots_adjust(top=0.9)
    fig2.suptitle("Training Summary ({date_str})")
    Episode = int(Episode_entry.get())
    save_path = create_save_path(Save_entry.get(), filename=f'reward_and_loss_{Episode}.png')
    fig2.savefig(save_path)
    return fig2
    
                               #############################################

###################### DISPLAY PLOTAMOR  ##############################


def show_figure_in_gui(fig, container_frame):
    for widget in container_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=container_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def run_test():
    MW_PLOT = tk.Toplevel(MW)
    MW_PLOT.title("เอเยน  Plot")
    MW_PLOT.geometry("1280x720")
    MW_PLOT.configure(bg="#08086A")

    plot_frame = tk.Frame(MW_PLOT)
    plot_frame.grid(row = 1 ,column = 0 )

    plot_frame_ANOTHER = tk.Frame(MW_PLOT)
    plot_frame_ANOTHER.grid(row = 2 ,column = 0 )
    
    fig = test_agent(env, Agent)
    show_figure_in_gui(fig, plot_frame)
  
    fig2 = saveplot(reward_step,Actor_loss,Critic_loss)
    show_figure_in_gui(fig2, plot_frame_ANOTHER)

def start_training():
    # อ่านค่าจาก GUI
    radius_str = radius_entry.get().strip()
    if not radius_str:
       messagebox.showerror("Invalid Input", "Please enter a value for radius.")
       return
    try:
        radius = float(radius_str)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number for radius.")
        return
        
    max_height = float(height_entry.get())
    max_flow = float(flow_entry.get())
    outlet_size = float(outlet_entry.get())
    tank_type = selected_tank.get()
    save_path = Save_entry.get()
    max_minutes = float(Max_entry.get())
    Episode = int(Episode_entry.get())  # <-- ต้องแปลงเป็น int ด้วย

    global plot_reward_Ep  # ถ้าจะใช้ต่อหลังจากฝึก
    plot_reward_Ep = []

    # เตรียม folder
    floder_name = 'Evaluation_Phase_DDPG_9_5'
    start_train = time.time()
    
    for Ep in range(Episode):
        progress_percent = (Ep + 1) / Episode * 100
        progress_var.set(progress_percent)
        
        Episode_status_label.config(text=f"Episode: {Ep+1}/{Episode}")
        MW.update_idletasks() 

        elapsed_seconds = int(time.time() - start_train)
        minutes, seconds = divmod(elapsed_seconds, 60)
        elapsed_time_var.set(f"Elapsed Time: {minutes:02d}:{seconds:02d}")
        MW.update_idletasks()

        global reward_step, Actor_loss, Critic_loss
        state = env.reset()
        total_reward_step = 0
        step = 0
        done = False
        Critic_loss = 0
        Actor_loss = 0
        reward_step = []
        loss_step_Critic = []
        loss_step_Actor = []
        plot_Action = []
        start_time = time.time()

        while not done:
            elapsed_time = time.time() - start_train
            action = Agent.select_action(state, True)
            next_state, reward, done, error, prev_error = env.step(action)
            Agent.memory.store(state, action, reward, next_state, done)
            state = next_state
            step += 1

            if step % 1000 == 0:
                Critic_loss, Actor_loss = Agent.Optimize_model()

            if (time.time() - start_time) > max_minutes * 60:
                print(f"Episode {Ep}: Training stopped due to time limit.")
                break

            total_reward_step += reward
            reward_step.append(reward)
            loss_step_Critic.append(Critic_loss)
            loss_step_Actor.append(Actor_loss)
            plot_Action.append(action)

        # บันทึก model และข้อมูล
        Agent.save_model(f"{floder_name}/model_auto_save", f'Auto_save_{Ep}.pth') 
        saveplot(reward_step, loss_step_Actor, loss_step_Critic)
        plot_reward_Ep.append(total_reward_step)

        # บันทึก CSV
        df = pd.DataFrame(reward_step)
        df2 = pd.DataFrame(loss_step_Actor)
        df3 = pd.DataFrame(loss_step_Critic)
        df4 = pd.DataFrame(plot_Action)

        df.to_csv(create_save_path(f"{floder_name}/logs/reward", episode=Ep, filename=f"reward_{Ep}.csv"), index=False, header=False)
        df2.to_csv(create_save_path(f"{floder_name}/logs/Actor", filename=f"loss_Actor_{Ep}.csv"), index=False, header=False)
        df3.to_csv(create_save_path(f"{floder_name}/logs/Critic", filename=f"loss_Critic_{Ep}.csv"), index=False, header=False)
        df4.to_csv(create_save_path(f"{floder_name}/logs/Action", filename=f"Action_{Ep}.csv"), index=False, header=False)

        test_agent(env, Agent, run_time=20, episode=Ep)
        Agent.Update_Actor_and_Critic()
    run_test()
                               ###################### DISPLAY PLOTAMOR  ##############################

###### CLEAR LINES comand ######################################################## BUFFER
def clear_lines(n=2):
    for _ in range(n):
        sys.stdout.write('\x1b[1A') 
        sys.stdout.write('\x1b[2K') 
    sys.stdout.flush()
                                       ##########################################################

update = 0
step_decay2 = 0
plot_reward_Ep = []
plot_loss_Ep = []
Decay_eps_step = 10000
start_trian = time.time()



Current_Q = 0
Target_Q = 0
floder_name = 'Evaluation_Phase_DDPG_9_5'


        # Save button to select save path
Save = tk.Label(MW, text="💾 Save File Path",font=("Arial",12))
Save.place(x=40, y=40)
Save_entry = tk.Entry(MW,font=("Arial",12),width=20)
Save_entry.place(x=180, y=40) 

########### plot frame

btn = ttk.Button(MW, text="Run Agent Test", command=threaded_training)
btn.place(x=670, y=40,width=100)
                      ############################



browse_button = tk.Button(MW, text="📂Browse", font=("Arial", 11), command=Browse_save_path)
browse_button.place(x=375, y=40)


tk.Button(MW, text="Create Path",font=("Arial", 11), command=on_create_path).place(x=470, y=40,width=100)


########################################################################


#tk.Label(SETENV, text="Type of Reward:",font=("Arial", 12)).grid(row=0, column=0, sticky="w")
#reward_type = ttk.Combobox(SETENV, values=["sparse", "dense", "custom"],width=17)
#reward_type.configure(font=("Arial", 12))
#reward_type.grid(row=0, column=1, padx=5, pady=15)


tk.Label(SETENV, text="Radius:",font=("Arial", 12)).grid(row=1, column=0, sticky="w")
radius_entry = tk.Entry(SETENV,font=("Arial", 12))
radius_entry.grid(row=1, column=1, padx=5, pady=20)

tk.Label(SETENV, text="Max Height of Tank:",font=("Arial", 12)).grid(row=2, column=0, sticky="w")
height_entry = tk.Entry(SETENV,font=("Arial", 12))
height_entry.grid(row=2, column=1, padx=5, pady=25)

tk.Label(SETENV, text="Max Flow Input:",font=("Arial", 12)).grid(row=3, column=0, sticky="w")
flow_entry = tk.Entry(SETENV,font=("Arial", 12))
flow_entry.grid(row=3, column=1, padx=5, pady=30)

tk.Label(SETENV, text="Size of Outlet Hole:",font=("Arial", 12)).grid(row=4, column=0, sticky="w")
outlet_entry = tk.Entry(SETENV,font=("Arial", 12))
outlet_entry.grid(row=4, column=1, padx=5, pady=35)

tk.Label(SETENV, text = "Episode:",font=("Arial", 12)).grid(row=5, column=0, sticky="w")
Episode_entry = tk.Entry( SETENV, font=("Arial", 12))
Episode_entry.grid(row=5, column=1, padx=5, pady=40)

tk.Label(SETENV, text = "Max minutes:",font=("Arial", 12)).grid(row=6, column=0, sticky="w")
Max_entry = tk.Entry( SETENV, font=("Arial", 12))
Max_entry.grid(row=6, column=1, padx=5, pady=40)


############# IMAGE

OGSILO_image = Image.open("SILO.png")
SILO_image = OGSILO_image.resize((50, 50))
silo_img = ImageTk.PhotoImage(SILO_image)

OGHORIZON_image = Image.open("HORIZON.png")
HORIZON_image = OGHORIZON_image.resize((50, 50))
cylinder_img = ImageTk.PhotoImage(HORIZON_image)

OGVETICAL_image = Image.open("VERTICAL.png")
VETICAL_image = OGVETICAL_image.resize((50, 50))
VERTICAL_img = ImageTk.PhotoImage(VETICAL_image)


# Create a save path entry
selected_tank = tk.StringVar()
selected_tank.set("silo")
selected_formula = {"formula": None}


############## สูตรเตี๋ยว
#def update_formula():
    #tank = selected_tank.get()
    #if tank == "silo":
    #    selected_formula["formula"] = lambda r, h: np.pi * r**2 * h  # Silo = πr²h
    #elif tank == "cylindrical":
    #    selected_formula["formula"] = lambda r, h: (np.pi * r**2 * h) / 2  # Horizontal approx.
    #elif tank == "vertical":
     #   selected_formula["formula"] = lambda r, h: np.pi * r**2 * h  # เหมือน Silo
    #print(f"Tank type selected: {tank} → Formula set.")


#tk.Label(SETENV, text="Tank Type:", font=("Arial", 12)).grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))
radio_frame = tk.Frame(SETENV)
radio_frame.grid(row=6, column=0, sticky="w", columnspan=2, pady=20)
 

#tk.Radiobutton( radio_frame, text="Silo Tank",image=silo_img,compound="top",variable=selected_tank,value="silo").grid(row=0, column=0, padx=20)

#tk.Radiobutton(
    #radio_frame,
    #text="Horizontal Tank",
    #compound="top",
    #variable=selected_tank,
    #value="cylindrical"
#).grid(row=0, column=1, padx=20)

#tk.Radiobutton(
    #radio_frame,
    #text="Vertical Tank",
    #image=VERTICAL_img,  
    #compound="top",
    #variable=selected_tank,
    #value="vertical"
#).grid(row=0, column=2, padx=20)


#####################################  
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(Status, orient="horizontal", length=500, mode="determinate", variable=progress_var)
progress_bar.grid(row=1, column=0, columnspan=2, pady=10)
progress_bar["maximum"] = 100  # เปอร์เซ็นต์

elapsed_time_var = tk.StringVar()
elapsed_time_var.set("Elapsed Time: 00:00")
elapsed_time_label = ttk.Label(Status, textvariable=elapsed_time_var)
elapsed_time_label.grid(row=2, column=0, columnspan=2)

Episode_status_label = tk.Label(MW, text="Episode: 0/0", font=("Arial", 12), fg="white", bg="#08086A")
Episode_status_label.grid(row=12, column=0, columnspan=2, pady=10)
######################


MW.mainloop()
    



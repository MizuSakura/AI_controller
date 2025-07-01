import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self):
       self.df = pd.DataFrame()
       self.current_path = Path(os.getcwd())

    def add_data_log(self, columns_name, data_list):  
        if len(columns_name) != len(set(columns_name)):
            raise ValueError("columns_name contains duplicate columns, please check your input")
        
        max_len = max(len(col_data) for col_data in data_list)

        padded_data = []
        for col_data in data_list:
            if len(col_data) < max_len:
                col_data = col_data + [None] * (max_len - len(col_data))
            else:
                col_data = col_data[:max_len]
            padded_data.append(col_data)

        new_data = {col: data for col, data in zip(columns_name, padded_data)}

        if self.df.empty:
            self.df = pd.DataFrame(new_data)
        else:
            extra_cols = [col for col in self.df.columns if col not in columns_name]

            for col in extra_cols:
                new_data[col] = [None] * max_len
            
            all_cols = list(self.df.columns) + [col for col in columns_name if col not in self.df.columns]

            
            new_rows = pd.DataFrame(new_data)
            new_rows = new_rows.reindex(columns=all_cols, fill_value=None)
            if not self.df.empty:
                new_rows = new_rows.astype(self.df.dtypes.to_dict())
            self.df = pd.concat([self.df, new_rows], ignore_index=True)
        
    def show_data(self):
        print(self.df)

    def result_column(self,column_name=None):
        if self.df.empty:
            print("No data to display")
            return None
        
        if column_name is None:
            return None
        elif column_name in self.df.columns:
            return self.df[column_name]
        else:
            print(f'column: {column_name} does not exist in the data')
            return
        

    def save_to_csv(self, file_name, folder_name=None,path_name = None):
        if not file_name.endswith('.csv'):
            file_name += '.csv'

        if path_name is not None:
            path_to_save = Path(path_name)
        else:
            path_to_save = self.current_path
        if folder_name is None:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            folder_to_save = path_to_save / today
        else:
            folder_to_save = path_to_save / folder_name
        print(f'path file : {folder_to_save}')
        folder_to_save.mkdir(parents=True, exist_ok=True)
      
        path_file = folder_to_save / file_name

        self.df.to_csv(path_file, index=False)
        print(f"Data saved to {path_file}")

    def load_csv(self,path_file):
        path = Path(path_file)
        if path.suffix.lower() == ".csv":
            try:
                self.df = pd.read_csv(path)
            except Exception as e:
                print("File not found. Please check the file path and file name.")
        else:
            raise ValueError("CSV file not found. Please check the file extension ")


# import random

# # Create Logger instance
# logger = Logger()

# # Simulate large data: 100,000 rows for Temperature and Humidity
# num_rows = 100_000
# temperature_data = [random.uniform(20, 40) for _ in range(num_rows)]
# humidity_data = [random.uniform(40, 80) for _ in range(num_rows)]

# # Add large dataset
# logger.add_data_log(["Temperature", "Humidity"], [temperature_data, humidity_data])
# print("✅ Added large Temperature and Humidity data.")

# # Add another large dataset with new columns
# pressure_data = [random.uniform(990, 1020) for _ in range(num_rows)]
# wind_speed_data = [random.uniform(0, 20) for _ in range(num_rows)]

# logger.add_data_log(["Pressure", "WindSpeed"], [pressure_data, wind_speed_data])
# print("✅ Added large Pressure and WindSpeed data.")

# # Show the shape of the DataFrame
# print(f"DataFrame shape: {logger.df.shape}")

# # Optionally show a small sample (to avoid flooding your terminal)
# print(logger.df.head())

# # Save large CSV
# logger.save_to_csv("large_test_log")
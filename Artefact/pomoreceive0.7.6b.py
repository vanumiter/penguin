import serial
import time
import csv
import statistics
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import mplcyberpunk # cool graphs



# processes temp csv file to get max, mean, and total sound during a work interval
def process_temp_data(temp_path):
    temp_df = pd.read_csv(temp_path)
    max_sound = temp_df.iloc[:,0].max()
    mean_sound = round(temp_df.iloc[:,0].mean(),2)
    return max_sound, mean_sound
    
# focus level input function
def focus_level_input():
    while True:
        try:
            focus_level = int(input("Rate your focus level throughout the work interval in a scale of 1-10: "))
            if 1 <= focus_level <= 10:
                return focus_level
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Error: Please enter an integer.")
            time.sleep(1)


# collecting and processing pomodoro data function
def collect_data():
    # open microbit serial port
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port = 'COM3'
    ser.open()

    # folder of csv files
    path = Path(__file__).parent.absolute()
    subdir = "csv"

    # temp csv file
    temp_csv_name = "pomotemp.csv"
    temp_path = str(path / subdir / temp_csv_name)
    temp_csvfile = open(temp_path, "w", newline='')  # clears temp csv file
    temp_csvfile.close()
    temp_csvfile = open(temp_path, "a", newline='')

    print("Data collection initiated. Begin pomodoro timer to collect data.")
    # continuously reads and collects data from microbit, writes to temp csv file
    while True:
        if ser.readable():
            data = ser.readline().decode("utf-8").strip() # decodes from byte to string, then truncates whitespace
            if data == 'exit':
                print("Exiting program.")
                exit()
            elif data.startswith("Distraction count: "):
                distraction_count = [int(i) for i in data.split() if i.isdigit()] # extracts digit from 'distraction count: xyz'
                distraction_count = distraction_count[0]
                print("Distraction count:", distraction_count)
                print("Pomodoro complete.")
                break
            elif len(data) > 0:
                print("Received:", data)
                temp_csvfile.write(data+"\n")
    temp_csvfile.close()
    ser.close()

    # process collected data
    max_sound, mean_sound = process_temp_data(temp_path)

    # get focus level
    focus_level = focus_level_input()

    # updates dictionary based on new values
    data_dict = {
        'mean_sound': mean_sound,
        'max_sound': max_sound,
        'distractions': distraction_count,
        'focus': focus_level
     }

    main_csv_name = "pomofinal.csv"
    main_dir = path / subdir / main_csv_name
    csv_file_path = str(main_dir) # set as string necessary, as the variable was classified as a directory path and not a string
    try: 
        csvfile = open(csv_file_path, "a", newline='')
        writer = csv.DictWriter(csvfile, fieldnames=['mean_sound', 'max_sound', 'distractions', 'focus']) # dictwriter calls the variables from the dictionary above and assigns them into their csv cells
        if main_dir.stat().st_size == 0: 
            writer.writeheader()
        writer.writerow(data_dict)
        csvfile.close()
    except Exception as e:
        print(f"Error writing to CSV: {e}")

    print("Data processing completed.")


def view_statistics():
    # csv file location
    path = Path(__file__).parent.absolute()
    subdir = "csv"
    main_csv_name = "pomofinal.csv"
    main_dir = path / subdir / main_csv_name
    csv_file_path = str(main_dir)

    # loading dataset and style
    plt.style.use("cyberpunk")
    data = pd.read_csv(csv_file_path) 

    # defining  four parameters and insight (target)
    x = data[['mean_sound', 'max_sound', 'distractions']]
    y = data['focus']

    # splitting dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(x_train, y_train) # fitting model with training data

    # predicts focus level based on user inputs using trained model 
    def predict_focus(mean_sound, max_sound, distractions):
        df = pd.DataFrame([[mean_sound, max_sound, distractions]], columns=['mean_sound', 'max_sound', 'distractions'])  
        return model.predict(df)[0]

    # produces scatter plot with linear regression trendline 
    def regression_graph(x, y):
        fig, axs = plt.subplots(1,3, figsize = (18,6))

        # loop through indexed list, graphing a subplot for each parameter
        for i,v in enumerate(['mean_sound', 'max_sound', 'distractions']):
            independent = x[v]
            
            # Fit a simple linear regression for plotting
            trend_model = LinearRegression()
            trend_model.fit(independent.values.reshape(-1, 1), y)

            # Making predictions for the trendline
            x_range = np.linspace(independent.min(), independent.max(), 100)
            y_pred = trend_model.predict(x_range.reshape(-1, 1))

            # Plotting
            axs[i].scatter(independent, y)
            axs[i].plot(x_range, y_pred)
            axs[i].set_xlabel(v)
            axs[i].set_ylabel('focus_level')
            axs[i].set_title(f'focus_level vs {v}')

            r, p = stats.pearsonr(independent, y)
            axs[i].text(0, 1, 'r = {:.2f}'.format(r), fontsize=11, transform=axs[i].transAxes)

            # glowy effects
            mplcyberpunk.add_glow_effects(axs[i])
            mplcyberpunk.add_gradient_fill(axs[i], alpha_gradientglow=0.5)
    
        
        plt.savefig("LinearRegression.png") # save chart to an image
        plt.tight_layout() # fit to screen
        # preview chart in Thonny
        plt.show()

    
    while True:
        choice_statistics = input("Choose which prediction you would like to see:\n1. What-if Question 1: Focus level in a quiet environment\n2. What-if Question 2: Focus level in a loud environment\n3. Prediction based on user input\n4. Visualize what-if questions as a bar chart\n5. Visualize focus level against each parameter as a graph\n6. Return to main menu\nEnter a number: ")
        if choice_statistics == '1':
            # hard coded values of a quiet environment based on csv data
            mean_quiet = 0.5
            max_quiet = 170
            distractions_quiet = 3

            quiet_focus = round(predict_focus(mean_quiet, max_quiet, distractions_quiet),2)
            
            print("On a scale of 1-10, the predicted focus level for a quiet environment is", quiet_focus, "\n")
            input("Press Enter to continue...")

        elif choice_statistics == '2':
            mean_loud = 16
            max_loud = 255
            distractions_loud = 13

            loud_focus = round(predict_focus(mean_loud, max_loud, distractions_loud),2)

            print("On a scale of 1-10, the predicted focus level for a loud environment is", loud_focus, "\n")
            input("Press Enter to continue...")

        elif choice_statistics == '3':
            # checks if sound levels are 0-255 and mean is less than max
            while True:
                try:
                    input_mean = float(input("Enter mean sound level: "))
                    input_max = float(input("Enter max sound level: "))
                    input_distractions = int(input("Enter number of distractions: "))

                    if input_mean > input_max:
                        print("Error: Mean sound level cannot be greater than max sound level.")
                    elif input_mean < 0 or input_mean > 255 or input_max < 0 or input_max > 255:
                        print("Error: Both mean and max sound levels must be between 0 and 255.")
                    else:
                        break
                except ValueError:
                    print("Error: Please enter an integer.")
                    time.sleep(1)

            # prediction based on user input
            predicted_focus = round(predict_focus(input_mean, input_max, input_distractions),2)
            if predicted_focus < 1:
                predicted_focus = 1
            elif predicted_focus > 10:
                predicted_focus = 10
            print("On a scale of 1-10, the predicted focus level for the values entered is", predicted_focus, "\n")
            input("Press Enter to continue...")

        elif choice_statistics == '4':
            # variables based off ones prior
            quiet_focus = predict_focus(0.5, 170, 3)  
            loud_focus = predict_focus(16, 250, 13)  

            # Environment names
            environments = ['Quiet', 'Loud']

            # Corresponding focus levels
            focus_levels = [quiet_focus, loud_focus]

            # Creating the bar chart
            bars = plt.bar(environments, focus_levels, color=['C0', 'C1'], zorder=2)
            plt.xlabel('Environment')
            plt.ylabel('Predicted Focus Level')
            plt.title('Predicted Focus Level by Environment')
            plt.ylim(0, 10)  # focus level is on a scale of 1-10
            
            # mplcyberpunk effects
            mplcyberpunk.add_bar_gradient(bars=bars)

            # mplcyberpunk effects
            mplcyberpunk.add_glow_effects()
            plt.savefig("WhatIfChart.png")
            plt.tight_layout()
            plt.show()

        elif choice_statistics == '5':
            print("Loading graphs...")

            regression_graph(x, y)
        elif choice_statistics == '6':
            print("Returning to main menu...")
            break 
        else:
            print("Error: Please enter a specified number.")
            time.sleep(1)

# interacting with program
print("Pomodoro Timer: Python Edition - Snapshot 23w46a\n")
while True:
    choice = input("Choose an option:\n1. Collect Pomodoro Timer Data\n2. View Statistics\n3. Exit\nEnter a number: ")
    if choice == '1':
        collect_data()
    elif choice == '2':
        view_statistics()
    elif choice == '3':
        print("Exiting...")
        break
    else:
        print("Error: Please enter a specified number.")
        time.sleep(1)
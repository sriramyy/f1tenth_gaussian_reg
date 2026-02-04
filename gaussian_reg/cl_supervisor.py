from .guassian import Model
from .params import DynamicParams, StaticParams
import numpy as np

def main():
    model = Model()
    
    # Try to load existing data
    model.load()

    print("[ ] Welcome to the command line supervisor")
    print("[ ] To use, enter either `I` to input parameters")
    print("    or `G` to get parameters or `T` to get and test parameters.")
    print("------------------------")
    
    while True:
        option = input("\nEnter option ([I]nput/[G]et/[T]est/[H]istory) or Q to quit: ").strip().upper()
        if option == "I":
            inputParams(model)
        elif option == "G":
            getParams(model)
        elif option == "T":
            # testing mode, so do both
            second_option = input("\nWhich type of testing? ([B]est/[E]xploration)").upper()
            testingMode(model, second_option)
        elif option == "H":
            historyMode(model)
        elif option == "Q":
            print("Exiting...")
            break
        else:
            print("Invalid option. Enter I, G, or Q.")


def historyMode(model:Model):
    "Prints the history of the model (Best lap or all laps)"

    option = input("\nEnter option ([B]est Lap, [A]ll laps): ").upper()
    if option == "B" or option == "":
        if len(model.y_history) == 0:
            print("No lap data recorded yet")
            return
        best_idx = np.argmin(model.y_history)
        best_laptime = model.y_history[best_idx]
        best_params = DynamicParams.from_array(model.X_history[best_idx])
        print(f"\n[BEST LAP] Laptime: {best_laptime:.4f}s , Params: {best_params.format()}")
    elif option == "A":
        print(f"Not implemented yet!")
    # TODO: implement the all laps printing

def testingMode(model:Model, option:str="Exploration"):
    """
    mode for testing so gets and inputs the params

    option  - "Exploration" - exploration-based testing
            - "Best" - tries to find best params
    """
    # first get the parameters to test
    print("Getting parameters to test...")
    if option.upper()=="EXPLORATION": type = ""
    if option.upper()=="BEST": type = "best"
    params_obj = model.request(type)

    # now test the parameters and get laptime from test
    print(f"Testing parameters: {params_obj.format()}")
    laptime = float(input("Enter laptime (s): "))

    model.record(params_obj, laptime)
    model.save()
    print("Data recorded successfully")

def inputParams(model:Model):
    try:
        print("Parameters: BUBBLE, MAX_LIDAR_DIST, STRAIGHT_SPEED, CORNER_SPEED, MAX_SPEED, STEER_SMOOTH_ALPHA, PREPROCESS_CONV_SIZE")
        params_raw = input("Enter the 7 parameters (comma separated): ")
        params = [float(p.strip()) for p in params_raw.split(",")]
        
        if len(params) != 7:
            print(f"Error: Expected 7 parameters, got {len(params)}.")
            return

        params_obj = DynamicParams.from_array(np.array(params))
        laptime = float(input("Enter laptime (s): "))
        
        model.record(params_obj, laptime)
        model.save()  # Auto-save after each recording
        print("Data recorded successfully.")
    except ValueError:
        print("Invalid input! Please enter numbers only.")


def getParams(model:Model):
    print("Getting parameters that the model suggests to test...")

    params_obj_explore = model.request()
    params_obj_best = model.request("best")

    print(f"\n Suggested EXPLORATION Params: {params_obj_explore.format()}")
    print(f" Suggested BEST Params: {params_obj_best.format()}")


if __name__ == "__main__":
    main()

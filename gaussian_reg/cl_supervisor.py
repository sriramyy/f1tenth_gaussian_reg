from .guassian import Model
from .params import DynamicParams, StaticParams
import numpy as np

def main():
    model = Model()
    
    # Try to load existing data
    model.load()

    print("[ ] Welcome to the command line supervisor")
    print("[ ] To use, enter either `I` to input parameters or")
    print("    or `G` to get parameters.")
    print("------------------------")
    
    while True:
        option = input("\nEnter option (I/G) or Q to quit: ").strip().upper()
        if option == "I":
            inputParams(model)
        elif option == "G":
            getParams(model)
        elif option == "Q":
            print("Exiting...")
            break
        else:
            print("Invalid option. Enter I, G, or Q.")


def inputParams(model:Model):
    try:
        print("Parameters: BUBBLE, MAX_LIDAR_DIST, STRAIGHT_SPEED, CORNER_SPEED, MAX_SPEED, MAX_STEER_ABS, STEER_SMOOTH_ALPHA, PREPROCESS_CONV_SIZE")
        params_raw = input("Enter the 8 parameters (comma separated): ")
        params = [float(p.strip()) for p in params_raw.split(",")]
        
        if len(params) != 8:
            print(f"Error: Expected 8 parameters, got {len(params)}.")
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

    params_obj = model.request()
    params = params_obj.to_array()
    # Integers: bubble_radius (0) and preprocess_conv_size (7)
    formatted = ", ".join([f"{p:.2f}" if i != 0 and i != 7 else str(int(p)) for i, p in enumerate(params)])
    print(f"\n Suggested Params: {formatted}")


if __name__ == "__main__":
    main()

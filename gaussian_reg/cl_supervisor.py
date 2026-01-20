from .guassian import Model
from .params import DynamicParams, StaticParams
import numpy as np

def main():
    model = Model()

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
    params_raw = input("Enter the parameters separated with commas (ex. 20, 1.0, 0.5...): ")

    # split the params into a list of params
    params = [float(p.strip()) for p in params_raw.split(",")]

    # convert into a dynamic params object
    params_obj = DynamicParams.from_array(np.array(params))

    # now need to get the lap time to input the params
    laptime_raw = input("Enter the laptime in seconds (ex. 10.3): ")
    laptime = float(laptime_raw)

    model.record(params_obj, laptime)

    print("Successfully recorded parameters and laptime into the model")


def getParams(model:Model):
    print("Getting parameters that the model suggests to test...")

    params_obj = model.request()
    params = params_obj.to_array()
    print(f"Suggested Parameters: {params}")


if __name__ == "__main__":
    main()

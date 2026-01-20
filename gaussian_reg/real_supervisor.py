from sklearn.metrics.pairwise import kernel_metrics, paired_euclidean_distances
from .guassian import Model
from .params import DynamicParams, StaticParams
import numpy as np
import time


def main():
    print("[ ] Starting Real Supervisor...")

    model = Model()

    print("[ ] Commands:")
    print("[ ] [ S ] Start Process, Sets (0,0)")
    print("[ ] [ R ] Car has been physically reset (0,0)")
    print("[ ] [ . ] ")

    while True:
        inp = input("Select option (S,R,)")
        if inp is "S":
            startProcess(model)
        elif inp is "R":
            print("")
            # TODO: implement this


def startProcess(model:Model):
    "Gets suggested params and runs FTG"
    
    # get and print the new params
    params = model.request()
    print("Testing the following parameters:")
    print(f"> {params.to_array()}")

    # we assume car is at (0,0) at this point
    start_time = time.perf_counter()
    start_location = getLocation()

    # run the follow the gap using these new params
    # function returns when reaches start point
    result = runFTG(params, start_location)

    # get the elapsed time
    if result:
        end_time = time.perf_counter()
        print(f"Lap completed, please return the car to an appropiate location if needed")
        elapsed_time = end_time - start_time
    else:
        elapsed_time = 100000
        # TODO: implement a way to manually return car to somewhere then start a new lap

    print(f"Elapsed Lap Time: {elapsed_time:.4f} s")
    model.record(params, elapsed_time)


def runFTG(params:DynamicParams, start_location:tuple) -> bool:
    "Runs the FTG algorithm and also returns when reaches start_location again"
    if getLocation() == start_location:
        return True

    # ... 
    # if crash:
    return False


def getLocation() -> tuple:
    return (0,0)

if __name__ == "__main__":
    main()

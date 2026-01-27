import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel
from .params import DynamicParams, StaticParams
from typing import cast
import pickle
import os

"""
Using a Gaussian Process Regressor Model to enhance parameter tuning in FTG algorithm where...
X - Parameters
y - Lap times 

"""

# TODO s:
#   - add a way to make FTG adjust params when its already running so i dont have to rebuild
#   - fix the steer value, keep that one constant and instead add like center bias as a dynamic one
#   - might need to clear the model data for it to work again
#   

class Model:
    def __init__(self, save_file:str = "model_data.pkl") -> None:
        # define bounds for each parameter
        # [bubble, lidar, straight_speed, corner_speed, speed_max, alpha, conv]
        self.bounds_min = np.array([20, 1.0, 1.0, 1.0, 1.0, 0.0, 1])
        self.bounds_max = np.array([100, 8.0, 8.0, 5.0, 7.0, 0.9, 10])

        # create the kernel which determines hwo smooth the funciton is
        kernel = C(1.0,(0.0001,10000)) * RBF(1.0,(0.001,1000))

        # create the regressor
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=0.01
        )

        # store history
        self.X_history:list[np.ndarray] = [] # list of np arrays
        self.y_history:list[float] = [] # list of laptimes
        
        # save file path
        self.save_file = save_file
        

    def record(self, params:DynamicParams, laptime:float) -> None:
        "Record the results of a lap"
        self.X_history.append(params.to_array())
        self.y_history.append(laptime)
        print(f"[MODEL] Learned lap time {laptime:.4f}s")
    
    
    def request(self) -> DynamicParams:
        "Request the best next params to test"
        
        # situation A: if in first x laps give a guess
        if len(self.X_history) < 3:
            random_arr = np.random.uniform(self.bounds_min, self.bounds_max)
            return DynamicParams.from_array(arr=random_arr)

        # situation B: train the model 
        X_train = np.array(self.X_history)
        y_train = np.array(self.y_history)
        self.gp.fit(X_train, y_train)

        # generate 1000 random candidates
        # might need to increase to 5k-10k if car not improving
        candidates = np.random.uniform(
            self.bounds_min, self.bounds_max, size=(1000,7)
        )
        
        # predict performance
        # mu = expected lap time, sigma = uncertainty
        mu, sigma = self.gp.predict(candidates, return_std=True)

        sigma = cast(np.ndarray, sigma)

        # acquisition function
        # score = mean - (exploration weight * uncertainty)
        # small mean (faster lap) and high uncertainty (explore more)
        # for exploraton weight::
        #   high - priortize testing params it doesn't know about
        #   low - focus on fine tuning params it already knows about
        exploration_weight = 1.0
        scores = mu - (exploration_weight * sigma)

        # pcik best
        # best meaning either very fast or very unknown 
        best_idx = np.argmin(scores)
        best_arr = candidates[best_idx]

        return DynamicParams.from_array(best_arr)
    
    
    def save(self) -> None:
        """Save the model's training history to disk"""
        data = {
            'X_history': self.X_history,
            'y_history': self.y_history
        }
        
        with open(self.save_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"[MODEL] Saved {len(self.X_history)} data points to {self.save_file}")
    
    
    def load(self) -> bool:
        """Load the model's training history from disk"""
        if not os.path.exists(self.save_file):
            print(f"[MODEL] No saved data found at {self.save_file}")
            return False
        
        try:
            with open(self.save_file, 'rb') as f:
                data = pickle.load(f)
            
            self.X_history = data['X_history']
            self.y_history = data['y_history']
            
            print(f"[MODEL] Loaded {len(self.X_history)} data points from {self.save_file}")
            
            # If we have enough data, refit the model
            if len(self.X_history) >= 3:
                X_train = np.array(self.X_history)
                y_train = np.array(self.y_history)
                self.gp.fit(X_train, y_train)
                print(f"[MODEL] Model refitted with loaded data")
            
            return True
        
        except Exception as e:
            print(f"[MODEL] Error loading data: {e}")
            return False

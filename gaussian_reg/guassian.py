import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel
from .params import DynamicParams, StaticParams
from typing import cast

"""
Using a Gaussian Process Regressor Model to enhance parameter tuning in FTG algorithm where...
X - Parameters
y - Lap times 

"""

class Model:
    def __init__(self) -> None:
        # define bounds for each parameter
        # [bubble, lidar, speed, steer, alpha, conv]
        self.bounds_min = np.array([20, 1.0, 1.0, 0.4, 0.0, 1])
        self.bounds_max = np.array([100, 8.0, 4.5, 1.0, 0.9, 10])

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
        

    def record(self, params:DynamicParams, laptime:float) -> None:
        "Record the results of a lap"
        self.X_history.append(params.to_array())
        self.y_history.append(laptime)
        print(f"[MODEL] Learned lap time {laptime:.4f}s")
    
    
    def request(self) -> DynamicParams:
        "Request the best next params to test"
        
        # Situation A: if in first x laps give a guess
        if len(self.X_history) < 5:
            random_arr = np.random.uniform(self.bounds_min, self.bounds_max)
            return DynamicParams.from_array(arr=random_arr)

        # Situation B: train the model 
        X_train = np.array(self.X_history)
        y_train = np.array(self.y_history)
        self.gp.fit(X_train, y_train)

        # generate 1000 random candidates
        candidates = np.random.uniform(
            self.bounds_min, self.bounds_max, size=(1000,6)
        )
        
        # predict performance
        # mu = expected lap time, sigma = uncertainty
        mu, sigma = self.gp.predict(candidates, return_std=True)

        sigma = cast(np.ndarray, sigma)

        # acquisition function
        # score = mean - (exploration weight * uncertainty)
        # small mean (faster lap) and high uncertainty (explore more)
        exploration_weight = 1.0
        scores = mu - (exploration_weight * sigma)

        # pcik best
        best_idx = np.argmin(scores)
        best_arr = candidates[best_idx]

        return DynamicParams.from_array(best_arr)

import numpy as np
import tensorflow as tf
# import tensorflow_privacy
# from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from sklearn.base import BaseEstimator, RegressorMixin
#  from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import tensorflow as tf
import collections
from accountant import *
from sanitizer import *
from pathlib import Path
import dp_accounting
import math
import json

import wfdb


ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
USE_PRIVACY = True
N_EPOCHS = 3


def random_batch(X, y, batch_size=64):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y.iloc[idx]

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# Function to find the appropriate noise multiplier (sigma_1)
def find_noise_multiplier(target_epsilon, delta, num_steps, sampling_probability, tol=0.01):
    low, high = 0.1, 10.0  # Initial range for sigma search
    while high - low > tol:
        mid = (low + high) / 2

        # Initialize RDP Accountant
        rdp_accountant = dp_accounting.rdp.RdpAccountant()

        event = dp_accounting.SelfComposedDpEvent(dp_accounting.PoissonSampledDpEvent(
            sampling_probability,
            dp_accounting.GaussianDpEvent(
                mid)), num_steps)

        # Apply DP-SGD mechanism
        rdp_accountant.compose(event)

        # Compute epsilon for given delta
        computed_epsilon = rdp_accountant.get_epsilon(delta)

        if computed_epsilon < target_epsilon:
            high = mid  # Reduce noise multiplier to decrease epsilon
        else:
            low = mid  # Increase noise multiplier to meet the target epsilon

    return round(mid, 3)



# Custom Scikit-learn Wrapper for Your Training Loop
class DPMLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, batch_size=64, learning_rate=0.001, learning_rate_z=0.001,
                 l2norm_bound=1.0, threshold_tau=1.0):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_z = learning_rate_z
        self.l2norm_bound = l2norm_bound
        self.threshold_tau = threshold_tau
        self.model = None  # Model will be created in fit()

    def build_model(self, input_shape):
        """Define the MLP model."""
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate), 
                      loss="mse", metrics=["mae"])

        return model

    def fit(self, X_train, y_train):
        """Train the model using the defined training loop."""
        self.model = self.build_model((1000, 12))

        eps = 1.0
        delta = 1e-9
        max_eps = 64.0
        max_delta = 1e-3

        # Given parameters
        dataset_size = 17418  # |D| 
        target_delta = 1 / (100 * dataset_size)  # delta = 5 * 10^{-7}
        target_eps = [0.5]  # Target epsilon values

        # Compute total number of training steps
        num_steps = (dataset_size // self.batch_size) * N_EPOCHS

        # Compute sampling probability
        sampling_probability = self.batch_size / dataset_size

        # Compute Noise Multiplier SIGMA_1
        sigma_1 = find_noise_multiplier(target_eps[0] / math.sqrt(2), target_delta, num_steps, sampling_probability, tol=0.01)
        
        C_0 = self.l2norm_bound / self.batch_size
        z_scb = C_0

        epsilon_1 = sigma_1 * C_0 * math.sqrt(num_steps)

        epsilon_2 = epsilon_1

        sigma_2 = epsilon_2 / math.sqrt(num_steps)

        '''
        optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
                        l2_norm_clip = self.l2norm_bound,
                        noise_multiplier = sigma_1,
                        num_microbatches = self.batch_size,
                        learning_rate = self.learning_rate)
                        '''
        optimizer = tf.optimizers.SGD(self.learning_rate)

        loss_fn = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
        ## threshold_tau=1.0
        ## learning_rate_z=0.001

        # Create accountant, sanitizer and metrics
        accountant = AmortizedAccountant(len(X_train))
        sanitizer = AmortizedGaussianSanitizer(accountant, [self.l2norm_bound / self.batch_size, True])

        n_steps = len(X_train) // self.batch_size
        spent_eps_delta = EpsDelta(0, 0)
        should_terminate = False

        for epoch in range(1, N_EPOCHS + 1):
            if should_terminate:
                spent_eps = spent_eps_delta.spent_eps
                spent_delta = spent_eps_delta.spent_delta
                print(f"Used privacy budget for {spent_eps:.4f}" +
                    f" eps, {spent_delta:.8f} delta. Stopping ...")
                break
            print(f"Epoch {epoch}/{N_EPOCHS}")
            for step in range(1, n_steps + 1):
                X_batch, y_batch = random_batch(X_train, y_train, self.batch_size)
                with tf.GradientTape() as tape:
                    y_pred = self.model(X_batch, training=True)
                    y_pred_final = tf.squeeze(y_pred[:, -1, :])
                    y_batch_reshaped = y_batch.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))
                    main_loss = tf.reduce_mean(loss_fn(y_batch_reshaped, y_pred_reshaped))     #####
                    loss = tf.add_n([main_loss] + self.model.losses)

                gradients = tape.gradient(loss, self.model.trainable_variables)

                if USE_PRIVACY:
                    sanitized_grads = []
                    eps_delta = EpsDelta(eps, delta)

                    # Adaptive Clipping Bound Calculation
                    b_t = sum([np.linalg.norm(g) > (self.threshold_tau * z_scb) for g in gradients])
                    b_t = b_t / self.batch_size + np.random.normal(0, sigma_2)
                    z_scb = z_scb * np.exp(self.learning_rate_z * b_t)  # Adaptive Clipping Update

                    for px_grad in gradients:
                        # Apply adaptive clipping with C0 as the base clipping bound
                        grad_norm = np.linalg.norm(px_grad)

                        if grad_norm > z_scb:
                            gamma_i = min(1, self.l2norm_bound / grad_norm)  # Compute scaling factor with C0
                        else:
                            gamma_i = min(1, self.l2norm_bound / z_scb)
                            # gamma_i = min(1, OPTION_C0 / grad_norm) * (OPTION_C0 / z_scb)

                        px_grad = gamma_i * px_grad  # Clip gradient

                        sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, sigma_1)
                        sanitized_grads.append(sanitized_grad)

                    spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]

                    ## grads_and_vars = optimizer._compute_gradients(loss, self.model.trainable_variables, tape = tape)
                    ## optimizer.apply_gradients(grads_and_vars)

                    optimizer.apply_gradients(zip(sanitized_grads, self.model.trainable_variables))

                    if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                        should_terminate = True
                else:
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return self

    def predict(self, X_test):
        """Make predictions on the test set."""
        return self.model.predict(X_test)

    def score(self, X_val, y_val):
        """Compute Mean Absolute Error (MAE) for GridSearchCV scoring."""
        y_pred = self.predict(X_val)
        y_pred_final = tf.squeeze(y_pred[:, -1, :])
        y_val_reshaped = y_val.values.reshape(-1, 1)
        y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))
        return -np.mean(np.abs(y_pred_reshaped - y_val_reshaped))  # Negative MAE (lower is better)



def main():

    path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    sampling_rate=100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Train
    X_train = X[np.where(Y.strat_fold < 9)]
    y_train = Y[(Y.strat_fold < 9)].age

    # Define the parameter grid for tuning
    param_grid = {
        "batch_size": [32, 64, 128],
        "learning_rate": [0.0001, 0.001, 0.01],
        "l2norm_bound": [0.1, 1.0, 5.0],
        "threshold_tau": [0.1, 1.0, 5.0],
        "learning_rate_z": [0.0001, 0.001, 0.01]

    }

    # Initialize the model
    dp_mlp = DPMLPRegressor()

    random_search = RandomizedSearchCV(dp_mlp, param_grid, n_iter=30, cv=3,
                                   scoring="neg_mean_absolute_error", verbose=2, n_jobs=-1, random_state=42)

    random_search.fit(X_train, y_train)

    # Get best parameters from GridSearchCV
    best_params = random_search.best_params_
    best_mae = -random_search.best_score_  # Convert back to positive MAE

    # Create a dictionary to save
    best_results = {
        "best_hyperparameters": best_params
    }

    # Save to JSON file
    with open("best_hyperparameters.json", "w") as f:
        json.dump(best_results, f, indent=4)

    # Output Best Hyperparameters
    print("Best Hyperparameters (RandomizedSearchCV):", best_params)
    print("Best MAE Score:", best_mae)


    """# Perform Grid Search
    grid_search = GridSearchCV(dp_mlp, param_grid, cv=3, scoring="neg_mean_absolute_error", verbose=2, n_jobs=-1)

    # Fit Grid Search to Training Data
    grid_search.fit(X_train, y_train)

    # Output Best Hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best MAE Score:", -grid_search.best_score_)"""

if __name__ == "__main__":
    main()
import numpy as np
import tensorflow as tf
import pandas as pd
import collections
from accountant import *
from sanitizer import *
from pathlib import Path
import dp_accounting
import math
import json
import wfdb
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

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


def build_regression_model(input_shape, learning_rate):
        """Define the MLP model."""
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), 
                      loss="mse", metrics=["mae"])

        return model


def train_model(batch_size, learning_rate_t, learning_rate_z, l2norm_bound, threshold_tau):
    """
    Train the model using the given hyperparameters and return the validation loss.
    """
    
    # Update hyperparameters
    BATCH_SIZE = batch_size
    LEARNING_RATE_T = learning_rate_t
    LEARNING_RATE_Z = learning_rate_z
    L2NORM_BOUND = l2norm_bound
    THRESHOLD_TAU = threshold_tau

    # Load dataset
    path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    sampling_rate=100

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Train
    X_train = X[np.where(Y.strat_fold < 9)]
    y_train = Y[(Y.strat_fold < 9)].age
    # Valid
    X_valid = X[np.where(Y.strat_fold == 9)]
    y_valid = Y[(Y.strat_fold == 9)].age


    # Build model
    model = build_regression_model((1000, 12), LEARNING_RATE_T)

    eps = 1.0
    delta = 1e-9
    max_eps = 64.0
    max_delta = 1e-3

    # Given parameters
    dataset_size = 17418  # |D| 
    target_delta = 1 / (100 * dataset_size)  # delta = 5 * 10^{-7}
    target_eps = [0.5]  # Target epsilon values

    # Compute total number of training steps
    num_steps = (dataset_size // BATCH_SIZE) * N_EPOCHS

    # Compute sampling probability
    sampling_probability = BATCH_SIZE / dataset_size

    # Compute Noise Multiplier SIGMA_1
    sigma_1 = find_noise_multiplier(target_eps[0] / math.sqrt(2), target_delta, num_steps, sampling_probability, tol=0.01)
    
    C_0 = L2NORM_BOUND / BATCH_SIZE
    z_scb = C_0

    epsilon_1 = sigma_1 * C_0 * math.sqrt(num_steps)

    epsilon_2 = epsilon_1

    sigma_2 = epsilon_2 / math.sqrt(num_steps)

    # Compile optimizer
    optimizer = tf.optimizers.SGD(LEARNING_RATE_T)
    loss_fn = tf.keras.losses.MeanSquaredError()

    optimizer = tf.optimizers.SGD(LEARNING_RATE_T)

    loss_fn = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    ## threshold_tau=1.0
    ## learning_rate_z=0.001

    # Create accountant, sanitizer and metrics
    accountant = AmortizedAccountant(len(X_train))
    sanitizer = AmortizedGaussianSanitizer(accountant, [L2NORM_BOUND / BATCH_SIZE, True])

    n_steps = len(X_train) // BATCH_SIZE
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
            X_batch, y_batch = random_batch(X_train, y_train, BATCH_SIZE)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                y_pred_final = tf.squeeze(y_pred[:, -1, :])
                y_batch_reshaped = y_batch.values.reshape(-1, 1)
                y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))
                main_loss = tf.reduce_mean(loss_fn(y_batch_reshaped, y_pred_reshaped))     #####
                loss = tf.add_n([main_loss] + model.losses)

            gradients = tape.gradient(loss, model.trainable_variables)

            if USE_PRIVACY:
                sanitized_grads = []
                eps_delta = EpsDelta(eps, delta)

                # Adaptive Clipping Bound Calculation
                b_t = sum([np.linalg.norm(g) > (THRESHOLD_TAU * z_scb) for g in gradients])
                b_t = b_t / BATCH_SIZE + np.random.normal(0, sigma_2)
                z_scb = z_scb * np.exp(LEARNING_RATE_Z * b_t)  # Adaptive Clipping Update

                for px_grad in gradients:
                    # Apply adaptive clipping with C0 as the base clipping bound
                    grad_norm = np.linalg.norm(px_grad)

                    if grad_norm > z_scb:
                        gamma_i = min(1, C_0 / grad_norm)  # Compute scaling factor with C0
                    else:
                        gamma_i = min(1, C_0 / z_scb)
                        # gamma_i = min(1, OPTION_C0 / grad_norm) * (OPTION_C0 / z_scb)

                    px_grad = gamma_i * px_grad  # Clip gradient

                    sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, sigma_1)
                    sanitized_grads.append(sanitized_grad)

                spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]

                ## grads_and_vars = optimizer._compute_gradients(loss, self.model.trainable_variables, tape = tape)
                ## optimizer.apply_gradients(grads_and_vars)

                optimizer.apply_gradients(zip(sanitized_grads, model.trainable_variables))

                if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                    should_terminate = True
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    X_batch, y_batch = random_batch(X_valid, y_valid)
    y_pred = model(X_batch, training=False)
    y_pred_final = tf.squeeze(y_pred[:, -1, :])                    #####

    y_batch_reshaped = y_batch.values.reshape(-1, 1)
    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

    main_loss = tf.reduce_mean(loss_fn(y_batch_reshaped, y_pred_reshaped))     #####
    loss = tf.add_n([main_loss] + model.losses)

    return loss  # Lower is better



def main():

    
    # Define hyperparameter search space
    space = [
        Integer(64, 512, name='batch_size'),            # Try larger batches for stability
        Real(0.0001, 0.02, name='learning_rate_t'),     # Smaller learning rate for DP-SGD
        Real(0.001, 0.1, name='learning_rate_z'),       # Adaptive clipping learning rate
        Real(1.0, 8.0, name='l2norm_bound'),            # Gradient clipping bound
        Real(1.0, 2.5, name='threshold_tau')            # Adaptive threshold
    ]

    @use_named_args(space)
    def objective(**params):
        return train_model(**params)

    # Run Bayesian Optimization
    res = gp_minimize(objective, space, n_calls=20, random_state=42)

    # Print best hyperparameters
    print("Best hyperparameters:", res.x)
    print(f"Best Validation Loss: {res.fun:.4f}")



if __name__ == "__main__":
    main()

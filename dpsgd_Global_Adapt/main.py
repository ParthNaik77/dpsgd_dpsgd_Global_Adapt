import numpy as np
import pandas as pd
import tensorflow as tf
import collections
import time
import matplotlib.pyplot as plt
from accountant import *
from sanitizer import *
from pathlib import Path
import dp_accounting

import wfdb
import ast


ROOT_DIR = Path(__file__).absolute().parent.parent
MODELS_DIR = Path.joinpath(ROOT_DIR, 'models')
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')

EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])
BATCH_SIZE = 64
LEARNING_RATE_T = 0.005     ## ETA_t
LEARNING_RATE_Z = 0.02     ## ETA_Z                ### Learning Rate for updating Z
L2NORM_BOUND = 3.0
## SIGMA_1 = 4.0
## SIGMA_2 = 1.0                                      ### Noise Multiplier for Threshold Noise
THRESHOLD_TAU = 1.2                               ### Threshold for adaptation
OPTION_C0 = L2NORM_BOUND / BATCH_SIZE
## z_scp = OPTION_C0                                  ### Strict Clipping Bound
DATASET = 'PTB_XL'
MODEL_TYPE = 'dense'
USE_PRIVACY = True
PLOT_RESULTS = True
N_EPOCHS = 100



def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
 
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

def main():
                    
    
    path = 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    sampling_rate=100
    z_scb = OPTION_C0                          ### Strict Clipping Bound

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    # Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)



    # Split data into train, valid and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold < 9)]
    y_train = Y[(Y.strat_fold < 9)].age

    # Valid
    X_valid = X[np.where(Y.strat_fold == 9)]
    y_valid = Y[(Y.strat_fold == 9)].age

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].age




    # Prepare network
    if MODEL_TYPE == 'dense':
        model = build_regression_model((1000, 12))
    else:
        # model = make_cnn_model((image_size, image_size, n_channels), num_classes)
        print('INVALID MODEL TYPE')
    loss_fn = tf.keras.losses.MeanSquaredError(name='mean_squared_error')
    optimizer = tf.optimizers.SGD(LEARNING_RATE_T)

    # Set constants for this loop
    eps = 1.0
    delta = 1e-7
    max_eps = 64.0
    max_delta = 1e-3
    ## target_eps = [64.0]
    ## target_delta = [1e-5] #unused


    # Given parameters
    dataset_size = 17418  # |D| 
    target_delta = 1 / (100 * dataset_size)  # delta = 5 * 10^{-7}
    target_eps = [1]  # Target epsilon values

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
    
    # Create accountant, sanitizer and metrics
    accountant = AmortizedAccountant(len(X_train))
    sanitizer = AmortizedGaussianSanitizer(accountant, [L2NORM_BOUND / BATCH_SIZE, True])
    
    # Setup metrics
    train_mean_loss = tf.keras.metrics.Mean(name = 'train_mean')
    valid_mean_loss = tf.keras.metrics.Mean(name = 'valid_mean')

    train_loss_scores, valid_loss_scores = list(), list()
    train_mae_scores, valid_mae_scores = list(), list()
    train_r2_scores, valid_r2_scores = list(), list()

    train_mae_metric = tf.keras.metrics.MeanAbsoluteError(name = 'train_mean_absolute_error')
    train_r2_metric = tf.keras.metrics.R2Score(name = 'train_r2_score')
    valid_mae_metric = tf.keras.metrics.MeanAbsoluteError(name = 'valid_mean_absolute_error')
    valid_r2_metric = tf.keras.metrics.R2Score(name = 'valid_r2_score')

    train_metrics = [tf.keras.metrics.MeanAbsoluteError(name = 'train_mean_absolute_error'), tf.keras.metrics.R2Score(name = 'train_r2_score')]
    valid_metrics = [tf.keras.metrics.MeanAbsoluteError(name = 'valid_mean_absolute_error'), tf.keras.metrics.R2Score(name = 'valid_r2_score')]
    test_metrics = [tf.keras.metrics.MeanAbsoluteError(name = 'test_mean_absolute_error'), tf.keras.metrics.R2Score(name = 'test_r2_score')]
    
    # Run training loop
    start_time = time.time()
    spent_eps_delta = EpsDelta(0, 0)
    should_terminate = False
    n_steps = len(X_train) // BATCH_SIZE
    for epoch in range(1, N_EPOCHS + 1):
        if should_terminate:
            spent_eps = spent_eps_delta.spent_eps
            spent_delta = spent_eps_delta.spent_delta
            print(f"Used privacy budget for {spent_eps:.4f}" +
                   f" eps, {spent_delta:.8f} delta. Stopping ...")
            break
        print(f"Epoch {epoch}/{N_EPOCHS}")
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train, y_train)
            with tf.GradientTape() as tape:
                y_pred = model(X_batch, training=True)
                y_pred_final = tf.squeeze(y_pred[:, -1, :])                  ##########


                y_batch_reshaped = y_batch.values.reshape(-1, 1)
                y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                main_loss = tf.reduce_mean(loss_fn(y_batch_reshaped, y_pred_reshaped))     #####
                loss = tf.add_n([main_loss] + model.losses)
                train_mae_metric.update_state(y_batch_reshaped, y_pred_reshaped)           #####
                train_r2_metric.update_state(y_batch_reshaped, y_pred_reshaped)

            gradients = tape.gradient(loss, model.trainable_variables)
            if USE_PRIVACY:
                sanitized_grads = []
                eps_delta = EpsDelta(eps, delta)

                # Adaptive Clipping Bound Calculation
                b_t = sum([np.linalg.norm(g) > (THRESHOLD_TAU * z_scb) for g in gradients])
                b_t = b_t / BATCH_SIZE + np.random.normal(0, sigma_2)
                z_scb = z_scb * np.exp(LEARNING_RATE_Z * b_t)  # Update strict clipping bound adaptively

                for px_grad in gradients:

                    # Apply adaptive clipping with C0 as the base clipping bound
                    grad_norm = np.linalg.norm(px_grad)

                    if grad_norm > z_scb:
                        gamma_i = min(1, C_0 / grad_norm)  # Compute scaling factor with C0
                    else:
                        gamma_i = min(1, C_0 / z_scb)

                    px_grad = gamma_i * px_grad  # Global Scaling

                    sanitized_grad = sanitizer.sanitize(px_grad, eps_delta, sigma_1)
                    sanitized_grads.append(sanitized_grad)

                spent_eps_delta = accountant.get_privacy_spent(target_eps=target_eps)[0]
                optimizer.apply_gradients(zip(sanitized_grads, model.trainable_variables))
                if (spent_eps_delta.spent_eps > max_eps or spent_eps_delta.spent_delta > max_delta):
                    should_terminate = True
            else:
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_mean_loss(loss)
            for metric in train_metrics:
                metric(y_batch_reshaped, y_pred_reshaped)                  #############
            if step % 200 == 0:
                time_taken = time.time() - start_time
                for metric in valid_metrics:
                    X_batch, y_batch = random_batch(X_valid, y_valid)
                    y_pred = model(X_batch, training=False)
                    y_pred_final = tf.squeeze(y_pred[:, -1, :])                    #####

                    y_batch_reshaped = y_batch.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    main_loss = tf.reduce_mean(loss_fn(y_batch_reshaped, y_pred_reshaped))     #####
                    loss = tf.add_n([main_loss] + model.losses)
                    valid_mean_loss(loss)
                    valid_mae_metric.update_state(y_batch_reshaped, y_pred_reshaped)           #####
                    valid_r2_metric.update_state(y_batch_reshaped, y_pred_reshaped) 
                    metric(y_batch_reshaped, y_pred_reshaped)                                  #####
                if USE_PRIVACY:
                    print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics, spent_eps_delta,) 
                else:
                    print_status_bar(step * BATCH_SIZE, len(y_train), train_mean_loss, time_taken,
                                     train_metrics + valid_metrics)
            if should_terminate:
                break
            
        # Update training scores
        train_mae = train_mae_metric.result()
        train_r2 = train_r2_metric.result()
        train_loss = train_mean_loss.result()

        train_mae_scores.append(train_mae)
        train_r2_scores.append(train_r2)
        train_loss_scores.append(train_loss)

        train_mae_metric.reset_state()   ##### changed from reset_states()
        train_r2_metric.reset_state()
        train_mean_loss.reset_state()    ##### changed from reset_states()
        
        # Update validation scores
        valid_mae = valid_mae_metric.result()
        valid_r2 = valid_r2_metric.result()
        valid_loss = valid_mean_loss.result()

        valid_mae_scores.append(valid_mae)
        valid_r2_scores.append(valid_r2)
        valid_loss_scores.append(valid_loss)

        valid_mae_metric.reset_state()
        valid_r2_metric.reset_state()
        valid_mean_loss.reset_state()
        
        if epoch % 10 == 0:
            for metric in test_metrics:
                y_pred = model(X_test, training=False)

                y_pred_final = tf.squeeze(y_pred[:, -1, :])     ################

                y_test_reshaped = y_test.values.reshape(-1, 1)
                y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                metric(y_test_reshaped, y_pred_reshaped)             #################
            metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                                  for m in test_metrics or []])
            print(f"Epoch {epoch} test accuracy: {metrics}")
    
    # Evaluate model
    for metric in test_metrics:
        y_pred = model(X_test, training=False)
        y_pred_final = tf.squeeze(y_pred[:, -1, :])       ###########

        y_test_reshaped = y_test.values.reshape(-1, 1)
        y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

        metric(y_test_reshaped, y_pred_reshaped)             ##########
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in test_metrics or []])
    print(f"Training completed, test metrics: {metrics}")
    ## print("test_metrics:::::::", train_mae_scores)
    
    # Save model
    version = "Target_eps_1_DPSGD-GLOBAL-ADAPT_MODEL_5" if USE_PRIVACY else "SGD"
    model.save(MODELS_DIR/f"{version}-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.h5")
    
    # Make plots
    if PLOT_RESULTS:
        epochs_range = range(1, N_EPOCHS+1)
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_loss_scores, color='blue', label='Training loss')
        plt.plot(epochs_range, valid_loss_scores, color='red', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Loss-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()
         
        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_mae_scores, color='blue', label='Training accuracy')
        plt.plot(epochs_range, valid_mae_scores, color='red', label='Validation accuracy')
        plt.title('Training and validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Mean Absolute Error-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()

        plt.figure(figsize=(8,6))
        plt.plot(epochs_range, train_r2_scores, color='blue', label='Training accuracy')
        plt.plot(epochs_range, valid_r2_scores, color='red', label='Validation accuracy')
        plt.title('Training and validation R2')
        plt.xlabel('Epochs')
        plt.ylabel('R2Score')
        plt.legend()
        plt.savefig(RESULTS_DIR/f"{version}-Coefficient of Determination-{N_EPOCHS}-{MODEL_TYPE}-{DATASET}.png")
        plt.close()



def build_regression_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.MeanSquaredError(), 
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.R2Score()])

    return model



def random_batch(X, y, batch_size=64):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y.iloc[idx]

def print_status_bar(iteration, total, loss, time_taken, metrics=None, spent_eps_delta=None):
    metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    if spent_eps_delta:
        spent_eps = spent_eps_delta.spent_eps
        spent_delta = spent_eps_delta.spent_delta
        print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
               f"{spent_eps:.4f}" + " - spent delta: " + f"{spent_delta:.8f}"
               " - time spent: " + f"{time_taken}" "\n", end=end)
    else:
        print("\r{}/{} - ".format(iteration, total) + metrics + " - spent eps: " +
              " - time spent: " + f"{time_taken}" "\n", end=end)

if __name__ == "__main__":
    main()

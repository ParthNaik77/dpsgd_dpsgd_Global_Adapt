import numpy as np
import pandas as pd
import tensorflow as tf
import collections
import time
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

import wfdb
import ast

ROOT_DIR = Path(__file__).absolute().parent.parent
RESULTS_DIR = Path.joinpath(ROOT_DIR, 'results')


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data



def main():

        path = 'dpsgd_Global_Adapt/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
        sampling_rate=100

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

        """y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
        print(f"y_test dtype: {y_test.dtype}") 

        y_test = tf.reshape(y_test, (-1, 1))"""

        model_DPSGD_GLOBAL_ADAPT_1 = tf.keras.models.load_model('models\MODEL_1_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_2 = tf.keras.models.load_model('models\MODEL_2_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_3 = tf.keras.models.load_model('models\MODEL_3_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_4 = tf.keras.models.load_model('models\MODEL_4_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_5 = tf.keras.models.load_model('models\MODEL_5_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_6 = tf.keras.models.load_model('models\MODEL_6_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_7 = tf.keras.models.load_model('models\MODEL_7_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_8 = tf.keras.models.load_model('models\MODEL_8_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_9 = tf.keras.models.load_model('models\MODEL_9_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_DPSGD_GLOBAL_ADAPT_10 = tf.keras.models.load_model('models\MODEL_10_HYP_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        
        
        model_DPSGD_1 = tf.keras.models.load_model('models\MODEL_1_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_2 = tf.keras.models.load_model('models\MODEL_2_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_3 = tf.keras.models.load_model('models\MODEL_3_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_4 = tf.keras.models.load_model('models\MODEL_4_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_5 = tf.keras.models.load_model('models\MODEL_5_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_6 = tf.keras.models.load_model('models\MODEL_6_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_7 = tf.keras.models.load_model('models\MODEL_7_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_8 = tf.keras.models.load_model('models\MODEL_8_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_9 = tf.keras.models.load_model('models\MODEL_9_HYP_DPSGD-100-dense-PTB_XL.h5')
        model_DPSGD_10 = tf.keras.models.load_model('models\MODEL_10_HYP_DPSGD-100-dense-PTB_XL.h5')
        

        model_target_eps_1_DPSGD_GLOBAL_ADAPT_1 = tf.keras.models.load_model('models\Target_eps_1_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_GLOBAL_ADAPT_2 = tf.keras.models.load_model('models\Target_eps_1_DPSGD-GLOBAL-ADAPT_MODEL_2-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_GLOBAL_ADAPT_3 = tf.keras.models.load_model('models\Target_eps_1_DPSGD-GLOBAL-ADAPT_MODEL_3-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_GLOBAL_ADAPT_4 = tf.keras.models.load_model('models\Target_eps_1_DPSGD-GLOBAL-ADAPT_MODEL_4-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_GLOBAL_ADAPT_5 = tf.keras.models.load_model('models\Target_eps_1_DPSGD-GLOBAL-ADAPT_MODEL_5-100-dense-PTB_XL.h5')

        model_target_eps_20_DPSGD_GLOBAL_ADAPT_1 = tf.keras.models.load_model('models\Target_eps_20_DPSGD-GLOBAL-ADAPT-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_GLOBAL_ADAPT_2 = tf.keras.models.load_model('models\Target_eps_20_DPSGD-GLOBAL-ADAPT_MODEL_2-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_GLOBAL_ADAPT_3 = tf.keras.models.load_model('models\Target_eps_20_DPSGD-GLOBAL-ADAPT_MODEL_3-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_GLOBAL_ADAPT_4 = tf.keras.models.load_model('models\Target_eps_20_DPSGD-GLOBAL-ADAPT_MODEL_4-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_GLOBAL_ADAPT_5 = tf.keras.models.load_model('models\Target_eps_20_DPSGD-GLOBAL-ADAPT_MODEL_5-100-dense-PTB_XL.h5')


        model_target_eps_1_DPSGD_1 = tf.keras.models.load_model('models\Target_eps_1_DPSGD-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_2 = tf.keras.models.load_model('models\Target_eps_1_DPSGD_MODEL_2-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_3 = tf.keras.models.load_model('models\Target_eps_1_DPSGD_MODEL_3-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_4 = tf.keras.models.load_model('models\Target_eps_1_DPSGD_MODEL_4-100-dense-PTB_XL.h5')
        model_target_eps_1_DPSGD_5 = tf.keras.models.load_model('models\Target_eps_1_DPSGD_MODEL_5-100-dense-PTB_XL.h5')

        model_target_eps_20_DPSGD_1 = tf.keras.models.load_model('models\Target_eps_20_DPSGD-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_2 = tf.keras.models.load_model('models\Target_eps_20_DPSGD_MODEL_2-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_3 = tf.keras.models.load_model('models\Target_eps_20_DPSGD_MODEL_3-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_4 = tf.keras.models.load_model('models\Target_eps_20_DPSGD_MODEL_4-100-dense-PTB_XL.h5')
        model_target_eps_20_DPSGD_5 = tf.keras.models.load_model('models\Target_eps_20_DPSGD_MODEL_5-100-dense-PTB_XL.h5')

        ## EPS 0.5
        models_dpsgd_ga = [model_DPSGD_GLOBAL_ADAPT_1,
                  model_DPSGD_GLOBAL_ADAPT_2,
                  model_DPSGD_GLOBAL_ADAPT_3,
                  model_DPSGD_GLOBAL_ADAPT_4,
                  model_DPSGD_GLOBAL_ADAPT_5,
                  model_DPSGD_GLOBAL_ADAPT_6,
                  model_DPSGD_GLOBAL_ADAPT_7,
                  model_DPSGD_GLOBAL_ADAPT_8,
                  model_DPSGD_GLOBAL_ADAPT_9,
                  model_DPSGD_GLOBAL_ADAPT_10]
        
        models_dpsgd = [model_DPSGD_1,
                        model_DPSGD_2,
                        model_DPSGD_3,
                        model_DPSGD_4,
                        model_DPSGD_5,
                        model_DPSGD_6,
                        model_DPSGD_7,
                        model_DPSGD_8,
                        model_DPSGD_9,
                        model_DPSGD_10]
        
        ## EPS 1.0
        models_dpsgd_ga_eps_1 = [model_target_eps_1_DPSGD_GLOBAL_ADAPT_1,
                                 model_target_eps_1_DPSGD_GLOBAL_ADAPT_2,
                                 model_target_eps_1_DPSGD_GLOBAL_ADAPT_3,
                                 model_target_eps_1_DPSGD_GLOBAL_ADAPT_4,
                                 model_target_eps_1_DPSGD_GLOBAL_ADAPT_5]
        
        models_dpsgd_eps_1 = [model_target_eps_1_DPSGD_1,
                              model_target_eps_1_DPSGD_2,
                              model_target_eps_1_DPSGD_3,
                              model_target_eps_1_DPSGD_4,
                              model_target_eps_1_DPSGD_5]
        
        ## EPS 20
        models_dpsgd_ga_eps_20 = [model_target_eps_20_DPSGD_GLOBAL_ADAPT_1,
                                 model_target_eps_20_DPSGD_GLOBAL_ADAPT_2,
                                 model_target_eps_20_DPSGD_GLOBAL_ADAPT_3,
                                 model_target_eps_20_DPSGD_GLOBAL_ADAPT_4,
                                 model_target_eps_20_DPSGD_GLOBAL_ADAPT_5]
        
        models_dpsgd_eps_20 = [model_target_eps_20_DPSGD_1,
                              model_target_eps_20_DPSGD_2,
                              model_target_eps_20_DPSGD_3,
                              model_target_eps_20_DPSGD_4,
                              model_target_eps_20_DPSGD_5]
        
        test_metrics = [tf.keras.metrics.MeanAbsoluteError(name = 'test_mean_absolute_error'), tf.keras.metrics.R2Score(name = 'test_r2_score')]



        mae_dpsgd_ga = []
        r2_dpsgd_ga = []
        mae_dpsgd = []
        r2_dpsgd = []

        mae_dpsgd_ga_eps_1 = []
        r2_dpsgd_ga_eps_1 = []
        mae_dpsgd_eps_1 = []
        r2_dpsgd_eps_1 = []
        
        mae_dpsgd_ga_eps_20 = []
        r2_dpsgd_ga_eps_20 = []
        mae_dpsgd_eps_20 = []
        r2_dpsgd_eps_20 = []


        ## EPS 0.5
        for model in models_dpsgd_ga:
            for metric in test_metrics:
                    y_pred = model(X_test, training=False)

                    y_pred_final = tf.squeeze(y_pred[:, -1, :])     

                    y_test_reshaped = y_test.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    metric(y_test_reshaped, y_pred_reshaped)

            mae_dpsgd_ga.append([test_metrics[0].result(), 'dpsgd_ga'])
            r2_dpsgd_ga.append([test_metrics[1].result(), 'dpsgd_ga'])
        
        for model in models_dpsgd:
            for metric in test_metrics:
                    y_pred = model(X_test, training=False)

                    y_pred_final = tf.squeeze(y_pred[:, -1, :])     

                    y_test_reshaped = y_test.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    metric(y_test_reshaped, y_pred_reshaped)

            mae_dpsgd.append([test_metrics[0].result(), 'dpsgd'])
            r2_dpsgd.append([test_metrics[1].result(), 'dpsgd'])

        dpsgd_mae_values = [float(tensor.numpy()) for tensor, _ in mae_dpsgd]
        dpsgd_ga_mae_values = [float(tensor.numpy()) for tensor, _ in mae_dpsgd_ga]

        dpsgd_r2_values = [float(tensor.numpy()) for tensor, _ in r2_dpsgd]
        dpsgd_ga_r2_values = [float(tensor.numpy()) for tensor, _ in r2_dpsgd_ga]

        ## EPS 1.0
        for model in models_dpsgd_ga_eps_1:
            for metric in test_metrics:
                    y_pred = model(X_test, training=False)

                    y_pred_final = tf.squeeze(y_pred[:, -1, :])     

                    y_test_reshaped = y_test.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    metric(y_test_reshaped, y_pred_reshaped)

            mae_dpsgd_ga_eps_1.append([test_metrics[0].result(), 'dpsgd_ga'])
            r2_dpsgd_ga_eps_1.append([test_metrics[1].result(), 'dpsgd_ga'])
        
        for model in models_dpsgd_eps_1:
            for metric in test_metrics:
                    y_pred = model(X_test, training=False)

                    y_pred_final = tf.squeeze(y_pred[:, -1, :])     

                    y_test_reshaped = y_test.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    metric(y_test_reshaped, y_pred_reshaped)

            mae_dpsgd_eps_1.append([test_metrics[0].result(), 'dpsgd'])
            r2_dpsgd_eps_1.append([test_metrics[1].result(), 'dpsgd'])

        eps_1_dpsgd_mae_values = [float(tensor.numpy()) for tensor, _ in mae_dpsgd_eps_1]
        eps_1_dpsgd_ga_mae_values = [float(tensor.numpy()) for tensor, _ in mae_dpsgd_ga_eps_1]

        eps_1_dpsgd_r2_values = [float(tensor.numpy()) for tensor, _ in r2_dpsgd_eps_1]
        eps_1_dpsgd_ga_r2_values = [float(tensor.numpy()) for tensor, _ in r2_dpsgd_ga_eps_1]

        ## EPS 20
        for model in models_dpsgd_ga_eps_20:
            for metric in test_metrics:
                    y_pred = model(X_test, training=False)

                    y_pred_final = tf.squeeze(y_pred[:, -1, :])     

                    y_test_reshaped = y_test.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    metric(y_test_reshaped, y_pred_reshaped)

            mae_dpsgd_ga_eps_20.append([test_metrics[0].result(), 'dpsgd_ga'])
            r2_dpsgd_ga_eps_20.append([test_metrics[1].result(), 'dpsgd_ga'])
        
        for model in models_dpsgd_eps_20:
            for metric in test_metrics:
                    y_pred = model(X_test, training=False)

                    y_pred_final = tf.squeeze(y_pred[:, -1, :])     

                    y_test_reshaped = y_test.values.reshape(-1, 1)
                    y_pred_reshaped = tf.reshape(y_pred_final, (-1, 1))

                    metric(y_test_reshaped, y_pred_reshaped)

            mae_dpsgd_eps_20.append([test_metrics[0].result(), 'dpsgd'])
            r2_dpsgd_eps_20.append([test_metrics[1].result(), 'dpsgd'])

        eps_20_dpsgd_mae_values = [float(tensor.numpy()) for tensor, _ in mae_dpsgd_eps_20]
        eps_20_dpsgd_ga_mae_values = [float(tensor.numpy()) for tensor, _ in mae_dpsgd_ga_eps_20]

        eps_20_dpsgd_r2_values = [float(tensor.numpy()) for tensor, _ in r2_dpsgd_eps_20]
        eps_20_dpsgd_ga_r2_values = [float(tensor.numpy()) for tensor, _ in r2_dpsgd_ga_eps_20]


        data_mae_eps_1_20 = {
        "Method": ["dpsgd"] * 5 + ["dpsgd_ga"] * 5 + ["dpsgd"] * 5 + ["dpsgd_ga"] * 5,
        "Target Epsilon": ["1"] * 10 + ["20"] * 10,
        "Value": (
            eps_1_dpsgd_mae_values +  # dpsgd @ 1
            eps_1_dpsgd_ga_mae_values +  # dpsgd_ga @ 1
            eps_20_dpsgd_mae_values +  # dpsgd @ 20
            eps_20_dpsgd_ga_mae_values   # dpsgd_ga @ 20
        )
        }

        # Create long-form DataFrame
        df_long = pd.DataFrame(data_mae_eps_1_20)

        # Plot grouped boxplot
        plt.figure(figsize=(10, 6))
        ## plt.ylim(14.5, 16.5)
        sns.boxplot(x="Target Epsilon", y="Value", hue="Method", data=df_long, palette="Set2")
        plt.title("MAE Boxplot Comparison of DPSGD vs DPSGD Global Adapt (Across Target Epsilons)")
        plt.xlabel("Target Epsilon")
        plt.ylabel("MAE")
        plt.legend(title="Method")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR/"BOX PLOT Mean Absolute Error Comparision EPS 1 vs EPS 20.png")
        plt.show()

        data_r2_eps_1_20 = {
        "Method": ["dpsgd"] * 5 + ["dpsgd_ga"] * 5 + ["dpsgd"] * 5 + ["dpsgd_ga"] * 5,
        "Target Epsilon": ["1"] * 10 + ["20"] * 10,
        "Value": (
            eps_1_dpsgd_r2_values +  # dpsgd @ 1
            eps_1_dpsgd_ga_r2_values +  # dpsgd_ga @ 1
            eps_20_dpsgd_r2_values +  # dpsgd @ 20
            eps_20_dpsgd_ga_r2_values   # dpsgd_ga @ 20
        )
        }

        # Create long-form DataFrame
        df_long = pd.DataFrame(data_r2_eps_1_20)

        # Plot grouped boxplot
        plt.figure(figsize=(10, 6))
        ## plt.ylim(14.5, 16.5)
        sns.boxplot(x="Target Epsilon", y="Value", hue="Method", data=df_long, palette="Set2")
        plt.title("R2 Score Boxplot Comparison of DPSGD vs DPSGD Global Adapt (Across Target Epsilons)")
        plt.xlabel("Target Epsilon")
        plt.ylabel("R2 Score")
        plt.legend(title="Method")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR/"BOX PLOT R2 Score Comparision EPS 1 vs EPS 20.png")
        plt.show()




        data_mae = {
        "Method": ["dpsgd"] * 10 + ["dpsgd_ga"] * 10,
        "Target Epsilon": ["0.5"] * 20,
        "Value": (
            dpsgd_mae_values +  # dpsgd @ 0.5
            dpsgd_ga_mae_values  # dpsgd_ga @ 0.5
        )
        }

        # Create long-form DataFrame
        df_long = pd.DataFrame(data_mae)

        # Plot grouped boxplot
        plt.figure(figsize=(10, 6))
        ## plt.ylim(14.5, 16.5)
        sns.boxplot(x="Target Epsilon", y="Value", hue="Method", data=df_long, palette="Set2")
        plt.title("MAE Boxplot Comparison of DPSGD vs DPSGD Global Adapt (Target Epsilons = 0.5)")
        plt.xlabel("Target Epsilon")
        plt.ylabel("MAE")
        plt.legend(title="Method")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR/"BOX PLOT Mean Absolute Error Comparision EPS 0.5.png")
        plt.show()

        data_r2 = {
        "Method": ["dpsgd"] * 10 + ["dpsgd_ga"] * 10,
        "Target Epsilon": ["0.5"] * 20,
        "Value": (
            dpsgd_r2_values +  # dpsgd @ 0.5
            dpsgd_ga_r2_values  # dpsgd_ga @ 0.5
        )
        }

        # Create long-form DataFrame
        df_long = pd.DataFrame(data_r2)

        # Plot grouped boxplot
        plt.figure(figsize=(10, 6))
        ## plt.ylim(14.5, 16.5)
        sns.boxplot(x="Target Epsilon", y="Value", hue="Method", data=df_long, palette="Set2")
        plt.title("R2 Score Boxplot Comparison of DPSGD vs DPSGD Global Adapt (Target Epsilons = 0.5)")
        plt.xlabel("Target Epsilon")
        plt.ylabel("R2 Score")
        plt.legend(title="Method")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR/"BOX PLOT R2 Score Comparision EPS 0.5.png")
        plt.show()




    

if __name__ == "__main__":
    main()

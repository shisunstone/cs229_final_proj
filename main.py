import keras.src.losses
import fastf1 as ff1
import fastf1.plotting as ff1plot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import parsing
import numpy as np
import sklearn
from scipy import special
from sklearn import metrics
import sklearn.model_selection
from tensorflow.keras import layers, models, regularizers


def parse_data(circuit):
    print('Parsing data!')
    X, Y_pit_int, Y_pit_num = parsing.parse_results(circuit)

    print('Making train-test split!')
    X_train, X_test, Y_pit_int_train, Y_pit_int_test, Y_pit_num_train, Y_pit_num_test = \
        sklearn.model_selection.train_test_split(X, Y_pit_int, Y_pit_num, test_size=0.25, shuffle=True)
    print('Finished making split!')

    return X_train, X_test, Y_pit_int_train, Y_pit_int_test, Y_pit_num_train, Y_pit_num_test


def implement_models(X_train, X_test, Y_pit_int_train, Y_pit_int_test, plot):
    n = Y_pit_int_train[0].shape[0]
    epochs_arr = np.arange(1, 11)
    cce_loss = keras.losses.CategoricalCrossentropy()

    # Setting up plotting
    if plot:
        fig, axs = plt.subplots(1, 1)
        fig.suptitle('Comparative training loss')
        axs.set_xlabel('Epoch Number')
        axs.set_ylabel('Loss')
        main_plot, axs_main = plt.subplots(2, 2)
        main_plot.suptitle('Comparative MAE')
        axs_main[0, 0].set_xlabel('Epoch Number')
        axs_main[0, 0].set_ylabel('F1 Score')
        axs_main[0, 1].set_xlabel('Epoch Number')
        axs_main[0, 1].set_ylabel('Accuracy (AUC ROC)')
        axs_main[1, 0].set_xlabel('Epoch Number')
        axs_main[1, 0].set_ylabel('Recall')
        axs_main[1, 1].set_xlabel('Epoch Number')
        axs_main[1, 1].set_ylabel('Precision')

    # -----------------------------------------------------------------------------------------------------------------#

    print('Buidling DNN!')
    # Define DNN architecture
    dnn = models.Sequential([
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(n, activation='softmax',
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     )
    ])
    print('Finished building DNN!')

    print('Compiling DNN!')
    # Compile the model
    dnn.compile(optimizer='adam', loss=cce_loss,
                metrics=[keras.metrics.F1Score(threshold=0.5, average='micro'),
                         keras.metrics.AUC(thresholds=[0.5]),
                         keras.metrics.Precision(thresholds=0.5),
                         keras.metrics.Recall(thresholds=0.5)])
    print('Finished compiling DNN!')

    print('Training DNN!')
    # Train the model
    dnn_losses = dnn.fit(X_train, Y_pit_int_train, epochs=10, batch_size=32)
    print('Finished training DNN!')

    if plot:
        axs.plot(epochs_arr, dnn_losses.history['loss'], label='DNN Loss')
        axs_main[0, 0].plot(epochs_arr, dnn_losses.history['f1_score'], label='DNN F1')
        axs_main[0, 1].plot(epochs_arr, dnn_losses.history['auc'], label='DNN Accuracy')
        axs_main[1, 0].plot(epochs_arr, dnn_losses.history['recall'], label='DNN Recall')
        axs_main[1, 1].plot(epochs_arr, dnn_losses.history['precision'], label='DNN Precision')

    print('Evaluating model!')
    # Evaluate the model on test set
    loss_dnn, mae_dnn, accuracy_dnn, recall_dnn, f1_score_dnn = dnn.evaluate(X_test, Y_pit_int_test)

    # -----------------------------------------------------------------------------------------------------------------#

    print('Reshaping data!')
    # Reshape input data for RNN
    X_timed_train, X_timed_test = np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1)

    print('Buidling RNN!')
    # Define RNN architecture
    rnn = models.Sequential([
        layers.SimpleRNN(32, activation='relu', input_shape=(1, X_timed_train.shape[2]),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(n, activation='softmax', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     )
    ])
    print('Finished building RNN!')

    print('Compiling RNN!')
    # Compile the model
    rnn.compile(optimizer='adam', loss=cce_loss,
                metrics=[keras.metrics.F1Score(threshold=0.5, average='micro'),
                         keras.metrics.AUC(thresholds=[0.5]),
                         keras.metrics.Precision(thresholds=0.5),
                         keras.metrics.Recall(thresholds=0.5)])
    print('Finished compiling RNN!')

    print('Training RNN!')
    # Train the model
    rnn_losses = rnn.fit(X_timed_train, Y_pit_int_train, epochs=10, batch_size=32)
    print('Finished training RNN!')

    if plot:
        axs.plot(epochs_arr, rnn_losses.history['loss'], label='RNN Loss')
        axs_main[0, 0].plot(epochs_arr, rnn_losses.history['f1_score'], label='RNN F1')
        axs_main[0, 1].plot(epochs_arr, rnn_losses.history['auc_1'], label='RNN Accuracy')
        axs_main[1, 0].plot(epochs_arr, rnn_losses.history['recall_1'], label='RNN Recall')
        axs_main[1, 1].plot(epochs_arr, rnn_losses.history['precision_1'], label='RNN Precision')

    print('Evaluating RNN!')
    # Evaluate the model (if desired)
    loss_rnn, mae_rnn, accuracy_rnn, recall_rnn, f1_score_rnn = rnn.evaluate(X_timed_test, Y_pit_int_test)

    # -----------------------------------------------------------------------------------------------------------------#

    print('Building LSTM model!')
    # Define LSTM architecture
    lstm = models.Sequential([
        layers.LSTM(32, activation='relu', input_shape=(1, X_timed_train.shape[2]),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(n, activation='sigmoid', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     )
    ])

    print('Compiling LSTM model!')
    # Compile the model
    lstm.compile(optimizer='adam', loss=cce_loss,
                 metrics=[keras.metrics.F1Score(threshold=0.5, average='micro'),
                          keras.metrics.AUC(thresholds=[0.5]),
                          keras.metrics.Precision(thresholds=0.5),
                          keras.metrics.Recall(thresholds=0.5)])

    print('Training LSTM model!')
    # Train the model
    lstm_losses = lstm.fit(X_timed_train, Y_pit_int_train, epochs=10, batch_size=32)
    print('Fimished training LSTM model!')
    if plot:
        axs.plot(epochs_arr, lstm_losses.history['loss'], label='LSTM Loss')
        axs_main[0, 0].plot(epochs_arr, lstm_losses.history['f1_score'], label='LSTM F1')
        axs_main[0, 1].plot(epochs_arr, lstm_losses.history['auc_2'], label='LSTM Accuracy')
        axs_main[1, 0].plot(epochs_arr, lstm_losses.history['recall_2'], label='LSTM Recall')
        axs_main[1, 1].plot(epochs_arr, lstm_losses.history['precision_2'], label='LSTM Precision')

    print('Evaluating LSTM model!')
    # Evaluate the model
    loss_lstm, mae_lstm, accuracy_lstm, recall_lstm, f1_score_lstm = lstm.evaluate(X_timed_test, Y_pit_int_test)

    # -----------------------------------------------------------------------------------------------------------------#

    print('Buidling GRU model!')
    # Define GRU architecture
    gru = models.Sequential([
        layers.GRU(32, activation='relu', input_shape=(1, X_timed_train.shape[2]),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     ),
        layers.Dense(n, activation='sigmoid', input_shape=(X_train.shape[1],),
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                     bias_regularizer=regularizers.l2(1e-4),
                     activity_regularizer=regularizers.l2(1e-5)
                     )
    ])

    print('Compiling GRU model!')
    # Compile the model
    gru.compile(optimizer='adam', loss=cce_loss,
                metrics=[keras.metrics.F1Score(threshold=0.5, average='micro'),
                         keras.metrics.AUC(thresholds=[0.5]),
                         keras.metrics.Precision(thresholds=0.5),
                         keras.metrics.Recall(thresholds=0.5)])
    print('Training GRU model!')
    # Train the model
    gru_losses = gru.fit(X_timed_train, Y_pit_int_train, epochs=10, batch_size=32)
    print(gru_losses.history.keys())
    if plot:
        axs.plot(epochs_arr, gru_losses.history['loss'], label='GRU Loss')
        axs_main[0, 0].plot(epochs_arr, gru_losses.history['f1_score'], label='GRU F1')
        axs_main[0, 1].plot(epochs_arr, gru_losses.history['auc_3'], label='GRU Accuracy')
        axs_main[1, 0].plot(epochs_arr, gru_losses.history['recall_3'], label='GRU Recall')
        axs_main[1, 1].plot(epochs_arr, gru_losses.history['precision_3'], label='GRU Precision')

    print('Evaluating GRU model!')
    # Evaluate the model (if desired)
    loss_gru, mae_gru, accuracy_gru, recall_gru, f1_score_gru = gru.evaluate(X_timed_test, Y_pit_int_test)

    # -----------------------------------------------------------------------------------------------------------------#
    print("The loss of the DNN is:", loss_dnn, "| And the MAE of the DNN is:", mae_dnn)
    print("The loss of the RNN is:", loss_rnn, "| And the MAE of the RNN is:", mae_rnn)
    print("The loss of the LSTM is:", loss_lstm, "| And the MAE of the LSTM is:", mae_lstm)
    print("The loss of the GRU is:", loss_gru, "| And the MAE of the GRU is:", mae_gru)

    if plot:
        axs.legend()
        axs_main[0, 0].legend(loc="lower left")
        axs_main[1, 0].legend(loc="lower left")
        axs_main[0, 1].legend(loc="lower left")
        axs_main[1, 1].legend(loc="lower left")
        plt.show()

    # Train Random Forest regressor
    rf_regressor = RandomForestRegressor(n_estimators=100)
    rf_regressor.fit(X_train, Y_pit_int_train)

    # Predict on test data (replace with your test data)
    predictions = np.array(rf_regressor.predict(X_test))
    proc_preds = np.zeros((predictions.shape[0], predictions.shape[1]))
    # Post-process predictions to get softmax one-hot vectors
    for i in range(predictions.shape[0]):
        proc_preds[i] = np.array(parsing.process_laps(predictions[i], 0.5))

    precision = keras.metrics.Precision()
    precision.update_state(Y_pit_int_test, proc_preds)
    print(precision.result())
    f1 = keras.metrics.F1Score(average='weighted')
    f1.update_state(Y_pit_int_test, proc_preds)
    print(f1.result())
    accuracy = keras.metrics.Accuracy()
    accuracy.update_state(Y_pit_int_test, proc_preds)
    print(accuracy.result())

    return dnn, rnn, lstm, gru, np.argmin([mae_dnn, mae_rnn, mae_lstm, mae_gru])


def predict_laps(year, driver, circuit, position, humidity, pressure, rain, tracktemp, windspeed, clear_cache=False,
                 plot=False):
    if clear_cache:
        ff1.Cache.clear_cache()
    circuit = circuit

    X_train, X_model_test, Y_pit_int_train, Y_pit_int_test, Y_pit_num_train, Y_pit_num_test = parse_data(circuit)

    dnn, rnn, lstm, gru, model_idx = implement_models(X_train, X_model_test, Y_pit_int_train, Y_pit_int_test, plot)

    num_laps = Y_pit_int_train.shape[1]
    test_year = year
    test_hum = humidity
    test_press = pressure
    test_rain = rain
    test_tt = tracktemp
    test_wind = windspeed

    if len(driver) == 3:
        code_bool = True
    else:
        code_bool = False

    test_prof = parsing.get_driver_proficiency(driver, coded=code_bool)
    test_circ = parsing.get_circuit_difficulty(circuit)
    test_pos = position

    X_test = np.array(
        [test_year, test_hum, test_press, test_rain, test_tt, test_wind, test_prof, test_circ, 1 / test_pos])
    Y_test = (parsing.parse_results_single(driver, circuit, year)).reshape(-1, 1)
    X_test_rnn = np.expand_dims(X_test, axis=1).reshape((-1, 1, X_test.shape[0]))

    X_test_dnn = X_test.reshape((-1, X_test.shape[0]))

    predictions_dnn = dnn.predict(X_test_dnn)[0]
    cm_dnn = metrics.confusion_matrix(Y_test, parsing.process_laps(predictions_dnn, 0.3))
    cm_dnn_d = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_dnn, display_labels=[False, True])
    cm_dnn_d.plot()
    cm_dnn_d.ax_.set_title("DNN Confusion Matrix")
    plt.show()
    pits_dnn = (-predictions_dnn).argsort()
    pits_dnn_1 = parsing.prune_laps(pits_dnn)

    predictions_rnn = rnn.predict(X_test_rnn)[0]
    cm_rnn = metrics.confusion_matrix(Y_test, parsing.process_laps(predictions_rnn, 0.3))
    cm_rnn_d = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_rnn, display_labels=[False, True])
    cm_rnn_d.plot()
    cm_rnn_d.ax_.set_title("RNN Confusion Matrix")
    plt.show()
    pits_rnn = (-predictions_rnn).argsort()
    pits_rnn_1 = parsing.prune_laps(pits_rnn)

    predictions_lstm = lstm.predict(X_test_rnn)[0]
    cm_lstm = metrics.confusion_matrix(Y_test, parsing.process_laps(predictions_lstm, 0.5))
    cm_lstm_d = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_lstm, display_labels=[False, True])
    cm_lstm_d.plot()
    cm_lstm_d.ax_.set_title("LSTM Confusion Matrix")
    plt.show()
    pits_lstm = (-predictions_lstm).argsort()
    pits_lstm_1 = parsing.prune_laps(pits_lstm)

    predictions_gru = gru.predict(X_test_rnn)[0]
    cm_gru = metrics.confusion_matrix(Y_test, parsing.process_laps(predictions_gru, 0.5))
    cm_gru_d = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_gru, display_labels=[False, True])
    cm_gru_d.plot()
    cm_gru_d.ax_.set_title("GRU Confusion Matrix")
    plt.show()
    pits_gru = (-predictions_gru).argsort()
    pits_gru_1 = parsing.prune_laps(pits_gru)

    pit_laps_lstm =  np.where(parsing.process_laps(predictions_lstm, 0.8) == 1)[0]
    pit_laps_gru = np.where(parsing.process_laps(predictions_gru, 0.8) == 1)[0]

    y_vals = np.concatenate((np.full(pit_laps_lstm.shape[0], 'LSTM'), np.full(pit_laps_gru.shape[0], 'GRU')))

    x_vals = np.concatenate((pit_laps_lstm, pit_laps_gru))

    plt.scatter(x_vals, y_vals)

    plt.show()

    return pits_dnn_1, pits_rnn_1, pits_lstm_1, pits_gru_1, model_idx, num_laps


def main():

    #  Input Test Information:
    test_year = 2022
    test_driver = 'SAI'
    test_circuit = 'Silverstone'
    test_position = 1
    test_hum = 70
    test_press = 1000
    test_rain = 0
    test_tt = 40
    test_wind = 5
    num_stops = 2

    dnn_pred, rnn_pred, lstm_pred, gru_pred, model_idx, num_laps = \
        predict_laps(test_year, test_driver, test_circuit, test_position,
                     test_hum, test_press, test_rain, test_tt, test_wind, clear_cache=False, plot=False)
    predictions = [dnn_pred, rnn_pred, lstm_pred, gru_pred]
    ideal_laps = list(predictions[2])
    ideal_stops = [ideal_laps[0]]
    while len(ideal_stops) < num_stops:
        for i in range(1, len(ideal_laps)):
            if abs(ideal_laps[i] - ideal_laps[0]) >= 10:
                lap = ideal_laps.pop(i)
                ideal_stops.append(lap)
                break

    fig, ax = plt.subplots(figsize=(10, 1.5))
    print("Number of laps:", num_laps)

    ideal_stops.append(num_laps)
    ideal_stops = sorted(ideal_stops)
    print(ideal_stops)
    previous_stint_end = 0
    for i in range(len(ideal_stops)):
        plt.barh(
            y=test_driver,
            width=ideal_stops[i] - previous_stint_end,
            left=previous_stint_end,
            color=ff1plot.DRIVER_COLORS[ff1plot.DRIVER_TRANSLATE[test_driver]],
            edgecolor="black",
            fill=True
        )

        previous_stint_end = ideal_stops[i]

    plt.title(f"{test_year} {test_circuit} LSTM Strategy for {test_driver}")
    plt.xlabel("Lap Number")
    plt.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

    ideal_laps = list(predictions[3])
    ideal_stops = [ideal_laps[0]]
    while len(ideal_stops) < num_stops:
        for i in range(1, len(ideal_laps)):
            if abs(ideal_laps[i] - ideal_laps[0]) >= 10:
                lap = ideal_laps.pop(i)
                ideal_stops.append(lap)
                break

    fig, ax = plt.subplots(figsize=(10, 1.5))
    print("Number of laps:", num_laps)

    ideal_stops.append(num_laps)
    ideal_stops = sorted(ideal_stops)
    print(ideal_stops)
    previous_stint_end = 0
    for i in range(len(ideal_stops)):
        plt.barh(
            y=test_driver,
            width=ideal_stops[i] - previous_stint_end,
            left=previous_stint_end,
            color=ff1plot.DRIVER_COLORS[ff1plot.DRIVER_TRANSLATE[test_driver]],
            edgecolor="black",
            fill=True
        )

        previous_stint_end = ideal_stops[i]

    plt.title(f"{test_year} {test_circuit} GRU Strategy for {test_driver}")
    plt.xlabel("Lap Number")
    plt.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

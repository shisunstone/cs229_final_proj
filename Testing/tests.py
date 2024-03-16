import numpy as np
import tensorflow as tf
import fastf1 as ff1
from matplotlib import pyplot as plt
import fastf1.plotting
import pandas as pd


def fastf1_test():
    # print(f1.get_event_schedule(2018))
    session = ff1.get_session(2018, 'germany', 'R')
    session.load()
    cur_weather = session.weather_data
    print(np.average(cur_weather['Humidity']), np.average(cur_weather['Pressure']), any(cur_weather['Rainfall']),
          np.average(cur_weather['TrackTemp']), np.average(cur_weather['WindSpeed']))
    # laps = session.laps
    #
    # drivers = session.drivers
    # print(drivers)
    #
    # drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]
    # print(drivers)
    #
    # stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    # stints = stints.groupby(["Driver", "Stint", "Compound"])
    # stints = stints.count().reset_index()
    #
    # stints = stints.rename(columns={"LapNumber": "StintLength"})
    # print(stints)
    #
    # fig, ax = plt.subplots(figsize=(5, 10))
    #
    # for driver in drivers:
    #     driver_stints = stints.loc[stints["Driver"] == driver]
    #
    #     previous_stint_end = 0
    #     for idx, row in driver_stints.iterrows():
    #         # each row contains the compound name and stint length
    #         # we can use these information to draw horizontal bars
    #         plt.barh(
    #             y=driver,
    #             width=row["StintLength"],
    #             left=previous_stint_end,
    #             color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]],
    #             edgecolor="black",
    #             fill=True
    #         )
    #
    #         previous_stint_end += row["StintLength"]


def main():
    # Generate some sample data with variable-length sequences
    num_samples = 1000
    max_num_laps = 50
    num_input_features = 8
    num_output_features = max_num_laps  # Assuming the same number of output features for all samples

    # Generate random input data with variable-length sequences
    input_data = [np.random.randn(np.random.randint(1, max_num_laps), num_input_features) for _ in range(num_samples)]

    # Generate random output data (probabilities of pitting)
    output_data = np.random.rand(num_samples, max_num_laps)

    # Define the RNN model
    class PitStopRNN(tf.keras.Model):
        def __init__(self):
            super(PitStopRNN, self).__init__()
            self.lstm = tf.keras.layers.LSTM(64, return_sequences=True)
            self.dense = tf.keras.layers.Dense(num_output_features, activation='sigmoid')

        def call(self, inputs):
            x = self.lstm(inputs)
            x = self.dense(x)
            return x

    # Instantiate the model
    model = PitStopRNN()

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Pad sequences to the same length
    padded_input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, padding='post', dtype='float32')

    # Train the model
    model.fit(padded_input_data, output_data, epochs=10, batch_size=32)

    # Example prediction
    # Generate a new sample input with variable-length sequence (replace with your actual data)
    new_input = [np.random.randn(np.random.randint(1, max_num_laps), num_input_features)]
    # Pad the new input sequence
    padded_new_input = tf.keras.preprocessing.sequence.pad_sequences(new_input, padding='post', dtype='float32')
    # Predict probabilities for pitting on each lap
    predictions = model.predict(padded_new_input)
    print("Predictions shape:", predictions.shape)
    print([predictions > 0.5])


if __name__ == "__main__":
    # df = pd.read_csv('../archive/circuits_full_stripped.csv')
    # x = np.arange(df.shape[0])
    # y = df['score']
    # y = np.sort(y)
    # plt.plot(x, y)
    # plt.show()
    fastf1_test()
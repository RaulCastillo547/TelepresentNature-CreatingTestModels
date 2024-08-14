import datetime
import math
import random

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf

MAX_EPOCS = 20
OUT_STEPS = 100

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 cv_name, label_columns=None):
        
        # Derive original data
        self.url_name = cv_name
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{cv_name}.csv')
        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])

        self.date_time = pd.to_datetime(self.orig_df.pop('timestamp'))
        self.orig_df.pop('external-temperature') # Since this class is for AI models that take in past position, non-positional inputs are dropped

        # If one position is desired, remove other position columns
        if label_columns == ['altitude']:
            self.orig_df.drop(columns=['longitude', 'latitude'], inplace=True)
        elif label_columns == ['longitude']:
            self.orig_df.drop(columns=['altitude', 'latitude'], inplace=True)
        elif label_columns == ['latitude']:
            self.orig_df.drop(columns=['altitude', 'longitude'], inplace=True)

        # Split data        
        n = len(self.orig_df)
        self.train_df = self.orig_df[0:int(n*0.7)]
        self.val_df = self.orig_df[int(n*0.7):int(n*0.9)]
        self.test_df = self.orig_df[int(n*0.9):]
        self.edge_df = self.test_df.iloc[-(input_width + shift):]
    
        # Normalize data
        self.train_mean = self.train_df.mean(numeric_only=True)
        self.train_std = self.train_df.std(numeric_only=True)
        
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std
        self.edge_df = (self.edge_df - self.train_mean) / self.train_std

        # Work out label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: index for index, name in enumerate(label_columns)}
        
        self.column_indices = {name: index for index, name in enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # Slicing input and label widths
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

    def extend_to_csv(self, url_dest, species, model=None, number_of_out_steps=1):
        # Load edge data
        base_df = self.orig_df
        add_on_data = []

        # Run model
        _, input_data = next(iter(self.edge_data))
        input_data = input_data.numpy()

        for i in range(number_of_out_steps):
            if (model is not None):
                input_data = model(input_data).numpy()
            else:
                input_data = np.array([input_data[-1][-1] for _ in range(self.label_width)]).reshape((1, 100, 3))
            
            add_on_data += input_data[0].tolist()
        
        # Define modeled and unmodeled datapoints
        base_df['timestamp'] = self.date_time
        base_df['modeled'] = False

        add_on_df = pd.DataFrame(add_on_data, columns=self.edge_df.columns)*self.train_std + self.train_mean

        start = max(self.date_time)
        add_on_time_index = list()

        assert species == 'Moose' or species == 'Deer'
        if (species == 'Moose'):
            for i in range(len(add_on_df)):
                add_on_time_index.append(start + datetime.timedelta(hours=3*(i+1)))
        elif (species == 'Deer'):
            for i in range(len(add_on_df)):
                add_on_time_index.append(start + datetime.timedelta(hours=4*(i+1)))
        
        add_on_df['timestamp'] = add_on_time_index
        add_on_df['modeled'] = True

        # Merge base_df and add_on_df and print out result
        full_df = pd.concat([base_df, add_on_df], ignore_index=True)
        full_df.set_index('timestamp', inplace=True)
        full_df.to_csv(f'CSVFiles/ExtendedCSV/{url_dest}_extended.csv', index=True, index_label='timestamp')

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:, self.column_indices[name]] for name in self.label_columns],
                axis = -1)
        
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col = 'altitude', max_subplots=3):
        inputs, labels = self.example
        
        plt.figure(figsize=(12, 8))
        
        plot_col_index = self.column_indices[plot_col]
        
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
    
    def make_dataset(self, data, n = None):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            seed=n,
            batch_size=32,)
        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def edge_data(self):
        return self.make_dataset(self.edge_df)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))

            self._example = result
        return result

class TemperatureWindow():
    def __init__(self, input_width, label_width, shift,
                 cv_name, label_columns=None):
        
        # Derive original data
        self.url_name = cv_name
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{cv_name}.csv')
        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])

        self.date_time = pd.to_datetime(self.orig_df.pop('timestamp'))
        self.orig_df.pop('longitude')
        self.orig_df.pop('latitude')
        self.orig_df.pop('altitude')

        # Split data        
        n = len(self.orig_df)
        self.train_df = self.orig_df[0:int(n*0.7)]
        self.val_df = self.orig_df[int(n*0.7):]
        self.test_df = self.orig_df[int(n*0.9):]
        self.edge_df = self.test_df.iloc[-(input_width + shift):]
    
        # Normalize data
        self.train_mean = self.train_df.mean(numeric_only=True)
        self.train_std = self.train_df.std(numeric_only=True)
        
        self.train_df = (self.train_df - self.train_mean) / self.train_std
        self.val_df = (self.val_df - self.train_mean) / self.train_std
        self.test_df = (self.test_df - self.train_mean) / self.train_std
        self.edge_df = (self.edge_df - self.train_mean) / self.train_std

        # Work out label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: index for index, name in enumerate(label_columns)}
        
        self.column_indices = {name: index for index, name in enumerate(self.train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        # Slicing input and label widths
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

    def create_df_extension(self, species, model=None, number_of_out_steps=1):
        # Load edge data
        base_df = self.orig_df
        add_on_data = []

        # Run model
        _, input_data = next(iter(self.edge_data))
        input_data = input_data.numpy()

        for i in range(number_of_out_steps):
            if (model is not None):
                input_data = model(input_data).numpy()
            else:
                input_data = np.array([input_data[-1][-1] for _ in range(self.label_width)]).reshape((1, 100, 1))
            
            add_on_data += input_data[0].tolist()
        
        # Define modeled and unmodeled datapoints
        base_df['timestamp'] = self.date_time
        base_df['modeled'] = False

        add_on_df = pd.DataFrame(add_on_data, columns=self.edge_df.columns)*self.train_std + self.train_mean

        start = max(self.date_time)
        add_on_time_index = list()

        assert species == 'Moose' or species == 'Deer'
        if (species == 'Moose'):
            for i in range(len(add_on_df)):
                add_on_time_index.append(start + datetime.timedelta(hours=3*(i+1)))
        elif (species == 'Deer'):
            for i in range(len(add_on_df)):
                add_on_time_index.append(start + datetime.timedelta(hours=4*(i+1)))
        
        add_on_df['timestamp'] = add_on_time_index
        add_on_df['modeled'] = True

        # Merge base_df and add_on_df and print out result
        full_df = pd.concat([base_df, add_on_df], ignore_index=True)
        full_df.set_index('timestamp', inplace=True)
        return full_df
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:, self.column_indices[name]] for name in self.label_columns],
                axis = -1)
        
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    
    def plot(self, model=None, plot_col = 'external-temperature', max_subplots=3):
        inputs, labels = self.example
        
        plt.figure(figsize=(12, 8))
        
        plot_col_index = self.column_indices[plot_col]
        
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                        label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
    
    def make_dataset(self, data, n = None):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            seed=n,
            batch_size=32,)
        ds = ds.map(self.split_window)

        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def edge_data(self):
        return self.make_dataset(self.edge_df)
    
    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))

            self._example = result
        return result


class RegressiveWindow():
    def __init__(self, csv_name):
        # Load and add month and day values
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{csv_name}.csv')
        
        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])
        self.orig_df['month'] = self.orig_df['timestamp'].map(lambda x: x.month)
        self.orig_df['day'] = self.orig_df['timestamp'].map(lambda x: x.day)
        self.timeline = self.orig_df.pop('timestamp')

        # Split data up
        self.train_df = self.orig_df.sample(frac=0.7, random_state=0)
        self.test_df = self.orig_df.drop(self.train_df.index)

        # Normalize data
        self.norm_train_df = (self.train_df - self.train_df.mean())/self.train_df.std()
        self.norm_test_df = (self.test_df - self.train_df.mean())/self.train_df.std()

        # Split input and labels
        self.train_input = self.norm_train_df[['external-temperature', 'month', 'day']].values
        self.test_input = self.norm_test_df[['external-temperature', 'month', 'day']].values

        self.train_label = self.norm_train_df[['longitude', 'latitude', 'altitude']].values
        self.test_label = self.norm_test_df[['longitude', 'latitude', 'altitude']].values

        # Reshape
        self.train_input = self.train_input.reshape((self.train_input.shape[0], 1, self.train_input.shape[1]))
        self.test_input = self.test_input.reshape((self.test_input.shape[0], 1, self.test_input.shape[1]))

        self.train_label = self.train_label.reshape((self.train_label.shape[0], 1, self.train_label.shape[1]))
        self.test_label = self.test_label.reshape((self.test_label.shape[0], 1, self.test_label.shape[1]))

        # Run autoregressive model for temperature predictions
        self.temperature_window = TemperatureWindow(input_width=OUT_STEPS, label_width=OUT_STEPS, shift=OUT_STEPS, cv_name=csv_name)
        self.autoregressive_model = FeedBack(units=32, out_steps=OUT_STEPS, num_vars=1)
        compile_and_fit(self.autoregressive_model, self.temperature_window)

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history

    def csv_extension(self, url_dest, species, model = None, out_steps = OUT_STEPS):
        # Define data for the add on dataframe
        add_on_data = {
            'timestamp': [],
            'external-temperature': [],
            'month': [],
            'day': [],
            'longitude': [],
            'latitude': [],
            'altitude': []
        }
        
        # Define function to get temperature values
        predicted_temp = self.temperature_window.create_df_extension(species, self.autoregressive_model, math.ceil(out_steps/self.temperature_window.label_width))

        for i in range(out_steps):
            # Get current date
            assert species == 'Moose' or species == 'Deer'
            if (species == 'Moose'):
                curr_date = max(self.timeline) + datetime.timedelta(hours=3*(i+1))
            elif (species == 'Deer'):
                curr_date = max(self.timeline) + datetime.timedelta(hours=4*(i+1))

            # Derive input variables
            external_temp = predicted_temp[predicted_temp['modeled'] == True].iloc[i, 0]
            month = curr_date.month
            day = curr_date.day

            # Retrieve position output
            if isinstance(model, tf.keras.Sequential):
                output_fields = model(np.array([(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]).reshape((1, 1, 3)))*self.train_df[['longitude', 'latitude', 'altitude']].std() + self.train_df[['longitude', 'latitude', 'altitude']].mean()
                output_fields = output_fields.numpy()[0][0]
            elif isinstance(model, KNeighborsRegressor):
                output_fields = model.predict([[(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]])[0]*self.train_df[['longitude', 'latitude', 'altitude']].std() + self.train_df[['longitude', 'latitude', 'altitude']].mean()
                output_fields = output_fields.values
            else:
                raise ValueError("Enter a Sequential Model or KNeighbors Regressor")

            longitude = output_fields[0]
            latitude = output_fields[1] 
            altitude = output_fields[2]

            # Load values
            add_on_data['timestamp'].append(curr_date)
            add_on_data['external-temperature'].append(external_temp)
            add_on_data['month'].append(curr_date.month)
            add_on_data['day'].append(curr_date.day)

            add_on_data['longitude'].append(longitude)
            add_on_data['latitude'].append(latitude)
            add_on_data['altitude'].append(altitude)
        
        # Generate base_df and add_on_df before combining them into one dataframe/csv
        base_df = self.orig_df.copy(deep=True)
        base_df['timestamp'] = self.timeline
        base_df['modeled'] = False
        
        add_on_df = pd.DataFrame(add_on_data)
        add_on_df['modeled'] = True
        
        combined_df = pd.concat([base_df, add_on_df], ignore_index=True)
        combined_df = combined_df[['timestamp', 'external-temperature', 'month', 'day', 'longitude', 'latitude', 'altitude', 'modeled']]

        combined_df.to_csv(f'CSVFiles/ExtendedCSV/{url_dest}_extended.csv', index=False)

class ClassificationWindow():
    def __init__(self, csv_name, n_clusters):
        # Load and Edit Pandas
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{csv_name}.csv')

        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])
        self.orig_df['month'] = self.orig_df['timestamp'].map(lambda x: x.month)
        self.orig_df['day'] = self.orig_df['timestamp'].map(lambda x: x.day)
        self.timeline = self.orig_df.pop('timestamp')

        # Split Data Up
        self.train_df = self.orig_df.sample(frac=0.9, random_state=0)
        self.test_df = self.orig_df.drop(self.train_df.index)

        # Normalize Data
        self.norm_train_df = (self.train_df - self.train_df.mean())/self.train_df.std()
        self.norm_test_df = (self.test_df - self.train_df.mean())/self.train_df.std()

        # Add Labels
        self.n_clusters = n_clusters
        train_pos = self.norm_train_df[['longitude', 'latitude', 'altitude']].values
        test_pos = self.norm_test_df[['longitude', 'latitude', 'altitude']].values

        self.k_means = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        train_labels = self.k_means.fit_predict(train_pos)
        test_labels = self.k_means.predict(test_pos)
        self.clusters = self.k_means.cluster_centers_

        self.norm_train_df['labels'] = train_labels
        self.norm_test_df['labels'] = test_labels

        # Split Input and Labels
        self.train_input = self.norm_train_df[['external-temperature', 'month', 'day']].values
        self.test_input = self.norm_test_df[['external-temperature', 'month', 'day']].values

        self.train_label = self.norm_train_df[['labels']].values
        self.test_label = self.norm_test_df[['labels']].values

        # Reshape
        self.train_input = self.train_input.reshape((self.train_input.shape[0], 1, self.train_input.shape[1]))
        self.test_input = self.test_input.reshape((self.test_input.shape[0], 1, self.test_input.shape[1]))

        self.train_label = self.train_label.reshape((self.train_label.shape[0], 1, self.train_label.shape[1]))
        self.test_label = self.test_label.reshape((self.test_label.shape[0], 1, self.test_label.shape[1]))

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history

    def csv_extension(self, url_dest, species, model=None, out_steps=OUT_STEPS):
        # Define data for the add on dataframe
        add_on_data = {
            'timestamp': [],
            'external-temperature': [],
            'month': [],
            'day': [],
            'longitude': [],
            'latitude': [],
            'altitude': [],
            'label': []
        }

        # Define function to get temperature values
        avg_temp = self.orig_df.groupby(['month', 'day']).mean()['external-temperature']

        for i in range(out_steps):
            # Get current date
            assert species == 'Moose' or species == 'Deer'
            if (species == 'Moose'):
                curr_date = max(self.timeline) + datetime.timedelta(hours=3*(i+1))
            elif (species == 'Deer'):
                curr_date = max(self.timeline) + datetime.timedelta(hours=4*(i+1))

            # Derive input variables
            external_temp = avg_temp[curr_date.month][curr_date.day]
            month = curr_date.month
            day = curr_date.day

            # Retrieve position output
            if isinstance(model, tf.keras.Sequential):
                prediction_field = model(np.array([(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]).reshape((1, 1, 3)))
                label = max(range(self.n_clusters), key = lambda x: prediction_field[0][0][x])
            elif isinstance(model, ClassificationBaseline):
                label = model.predict(np.array([(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                                (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                                (day - self.train_df.mean()['day'])/self.train_df.std()['day']]).reshape((1, 1, 3)))[0][0][0]
            else:
                raise ValueError("Enter a Sequential Model or Classification Baseline")

            point = self.clusters[label]*self.train_df[['longitude', 'latitude', 'altitude']].std() + self.train_df[['longitude', 'latitude', 'altitude']].mean()
            
            add_on_data['timestamp'].append(curr_date)
            add_on_data['external-temperature'].append(external_temp)
            add_on_data['month'].append(curr_date.month)
            add_on_data['day'].append(curr_date.day)

            add_on_data['longitude'].append(point[0])
            add_on_data['latitude'].append(point[1])
            add_on_data['altitude'].append(point[2])
            add_on_data['label'].append(label)
        
        # Generate base_df and add_on_df before combining them into one dataframe/csv
        add_on_df = pd.DataFrame(add_on_data)
        add_on_df['modeled'] = True

        base_df = self.orig_df.copy(deep=True)
        base_df['timestamp'] = self.timeline
        base_df['modeled'] = False

        combined_df = pd.concat([base_df, add_on_df], ignore_index=True)

        combined_df = combined_df[['timestamp', 'month', 'day', 'external-temperature', 'longitude', 'latitude', 'altitude','modeled']]

        combined_df.to_csv(f'CSVFiles/ExtendedCSV/{url_dest}_extended.csv', index=False)

class RNNWindow():
    def __init__(self, csv_name):
        # Load and Edit Pandas DF
        self.orig_df = pd.read_csv(f'CSVFiles/CleanCSV/{csv_name}.csv')
        self.orig_df['timestamp'] = pd.DatetimeIndex(self.orig_df['timestamp'])
        self.orig_df['month'] = self.orig_df['timestamp'].map(lambda x: x.month)
        self.orig_df['day'] = self.orig_df['timestamp'].map(lambda x: x.day)
        self.timeline = self.orig_df.pop('timestamp')

        # Split Data Up
        n = len(self.orig_df)
        self.train_df = self.orig_df[0:int(n*0.7)]
        self.test_df = self.orig_df[int(n*0.7):]

        # Normalize Data
        self.norm_train_df = (self.train_df - self.train_df.mean())/self.train_df.std()
        self.norm_test_df = (self.test_df - self.train_df.mean())/self.train_df.std()

        # Split Input and Labels
        self.train_input, self.train_label = self.split_data(self.norm_train_df, OUT_STEPS)
        self.test_input, self.test_label = self.split_data(self.norm_test_df, OUT_STEPS)

    def split_data(self, df, time_steps):
        assert len(df) >= time_steps
        input_sample = []
        output_sample = []

        for index in range(0, len(df) - time_steps, time_steps):
            section = df[index:index+time_steps]
            input_sample.append(section[['external-temperature', 'month', 'day']].values)
            output_sample.append(section[['longitude', 'latitude', 'altitude']].values)

        input_sample = np.array(input_sample)
        output_sample = np.array(output_sample)

        return input_sample, output_sample

    def model_compilation_and_fitting(self, model, patience=2):
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
        
        history = model.fit(self.train_input, self.train_label, epochs=MAX_EPOCS,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')])
        
        return history

    def csv_extension(self, url_dest, species, model = None, out_steps = OUT_STEPS):
        data = {
            'timestamp': [],
            'longitude': [],
            'latitude': [],
            'altitude': [],
            'external-temperature': [],
            'month': [],
            'day': []
        }
        
        avg_temp = self.orig_df.groupby(['month', 'day']).mean()['external-temperature']

        for i in range(out_steps):
            assert species == 'Moose' or species == 'Deer'
            if (species == 'Moose'):
                curr_date = max(self.timeline) + datetime.timedelta(hours=3*(i+1))
            elif (species == 'Deer'):
                curr_date = max(self.timeline) + datetime.timedelta(hours=4*(i+1))

            external_temp = avg_temp[curr_date.month][curr_date.day]
            month = curr_date.month
            day = curr_date.day

            output_fields = model(np.array([(external_temp - self.train_df.mean()['external-temperature'])/self.train_df.std()['external-temperature'], 
                                            (month - self.train_df.mean()['month'])/self.train_df.std()['month'], 
                                            (day - self.train_df.mean()['day'])/self.train_df.std()['day']]).reshape([1, 1, 3]))*self.train_df[['longitude', 'latitude', 'altitude']].std() + self.train_df[['longitude', 'latitude', 'altitude']].mean()

            output_fields = output_fields.numpy()[0][0]

            longitude = output_fields[0]
            latitude = output_fields[1] 
            altitude = output_fields[2]

            data['longitude'].append(longitude)
            data['latitude'].append(latitude)
            data['altitude'].append(altitude)
            data['external-temperature'].append(external_temp)
            data['month'].append(curr_date.month)
            data['day'].append(curr_date.day)
            data['timestamp'].append(curr_date)
        
        add_on_df = pd.DataFrame(data)
        add_on_df['modeled'] = True

        base_df = self.orig_df.copy(deep=True)
        base_df['timestamp'] = self.timeline
        base_df['modeled'] = False
        
        combined_df = pd.concat([base_df, add_on_df], ignore_index=True)

        combined_df = combined_df[['timestamp', 'month', 'day', 'external-temperature', 'longitude', 'latitude', 'altitude', 'modeled']]

        combined_df.to_csv(f'CSVFiles/ExtendedCSV/{url_dest}_extended.csv', index=False)

@tf.keras.utils.register_keras_serializable()
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_vars):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.num_vars = num_vars
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(self.num_vars)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state
    
    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the LSTM state.
        prediction, state = self.warmup(inputs)

        # Insert the first prediction.
        predictions.append(prediction)

        # Run the rest of the prediction steps.
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                    training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output.
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

    def get_config(self):
        config = dict()
        config.update({'units': self.units, 'out_steps': self.out_steps, 'num_vars': self.num_vars})
        return config

class ClassificationBaseline():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def predict(self, inputs):
        random_arr = np.array(random.choices(range(self.n_clusters), k=len(inputs))).reshape(inputs.shape[0], 1, 1)
        return random_arr

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])
    
    history = model.fit(window.train, epochs=MAX_EPOCS,
                        validation_data=window.val,
                        callbacks=[early_stopping])

    return history


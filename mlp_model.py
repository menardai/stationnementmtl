import sys
import datetime
import json

import numpy as np
import keras.backend as K

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import TensorBoard

from sklearn.externals import joblib


from dataset import Dataset


class MLPModel(object):
    """
    Multi-Layer Perceptron Model for predicting parking occupancy.
    """

    def __init__(self, input_layer_dim=8, unit_per_layer=700, hidden_layer_count=5,
                 pre_trained_model=False, verbose=False):
        """
        Build the model according the given parameters.
        """
        self.input_layer_dim = input_layer_dim
        self.unit_per_layer = unit_per_layer
        self.hidden_layer_count = hidden_layer_count

        # the feature scaler will be defined at training or loading time since it's based on the training set values
        self.feature_scaler = None

        if pre_trained_model:
            self._load(verbose)

        else:
            # Initialising the model
            self.model = Sequential()

            # Adding the input layer and the first hidden layer
            self.model.add(Dense(units=unit_per_layer, kernel_initializer='uniform', activation='relu', input_dim=input_layer_dim))

            for i in range(hidden_layer_count - 1):
                self.model.add(Dense(units=unit_per_layer, kernel_initializer='uniform', activation='relu'))

            # Adding the output layer
            self.model.add(Dense(units=5, kernel_initializer='uniform', activation='softmax'))

    def train(self, dataset, lr, batch_size, nb_epoch, save_weights=False, verbose=False):
        """
        Train the model using the given dataset.

        Arguments:
            dataset -- array (n, 8) : [DayOfYear, DayOfWeek, NoHeure, CenterLongitude, CenterLatitude, EmplacementCount, Temperature, Precip_total_day_mm]

        """
        experiment_name = '{}L_{}U__lr_{}_batch_{}'.format(self.hidden_layer_count, self.unit_per_layer, lr, batch_size)
        if verbose:
            print('Running experiment {}'.format(experiment_name))

        tbCallBack = TensorBoard(log_dir='./logs/'+experiment_name,
                                 histogram_freq=0, write_graph=False, write_images=False)

        # Compiling the ANN
        adam = optimizers.Adam(lr)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(dataset.X_train, dataset.Y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=[tbCallBack])
        self.feature_scaler = dataset.sc

        if save_weights:
            self._save()

    def predict(self, X):
        """
        Make prediction for the given date at every hours for all sections.

        Returns:
            y_pred -- numpy array of shape (n, 5), predicted occupancy rates
        """
        X = self.feature_scaler.transform(X)
        y_pred = self.model.predict(X)

        return y_pred

    def predict_sample(self, day_of_year, day_of_week, hour, section_longitude,
                       section_lattitude, section_parking_count, temperature, day_rain_mm):
        """
        Make prediction for the given sample.

        Returns:
            y_pred -- numpy array of shape (1, 5), predicted occupancy rate
        """
        X = np.array([day_of_year, day_of_week, hour, section_longitude,
                      section_lattitude, section_parking_count, temperature,
                      day_rain_mm]).reshape(1, 8)
        X = self.feature_scaler.transform(X)
        y_pred = self.model.predict(X)

        return y_pred

    def _load(self, verbose=False):
        """ Load serialized weights from HDF5 file and the feature scaler used during training. """
        architecture_filename = 'parking_model_{}x{}'.format(self.hidden_layer_count, self.unit_per_layer)

        json_file = open(architecture_filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        if verbose:
            print('Loading model json: {}'.format(architecture_filename + '.json'))
        self.model = model_from_json(loaded_model_json)

        if verbose:
            print('Loading model weights: {}'.format(architecture_filename + '.h5'))
        self.model.load_weights(architecture_filename + '.h5')

        if verbose:
            print('Loading training feature scaler: {}'.format(architecture_filename + '.scaler'))
        self.feature_scaler = joblib.load(architecture_filename + '.scaler')

    def _save(self):
        """ Serialize weights to HDF5 and the feature scaler used during training (if any). """
        architecture_filename = 'parking_model_{}x{}'.format(self.hidden_layer_count, self.unit_per_layer)

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(architecture_filename + '.json', "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save_weights(architecture_filename + '.h5')

        # save feature scaler (generated by Dataset class)
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, architecture_filename + '.scaler')

# ---------------------------------------------------------


def run_experiments():
    # model_list = [
    #     {'units':  100, 'layers':  2, 'lr': 0.0001, 'batch_size': 500, 'epochs': 100},
    #     {'units':  200, 'layers':  2, 'lr': 0.001,  'batch_size': 500, 'epochs': 100}
    # ]

    model_list = []
    for batch_size in [1000]:
        for lr in [0.0010, 0.0005, 0.0007]:
            for units in [700, 500]:
                for layers in [5, 7]:
                    model_list.append({'units':units, 'layers':layers, 'lr':lr, 'batch_size':batch_size, 'epochs': 150})

    print('Loading dataset...')
    dataset = Dataset('training_set.csv')

    for m in model_list:
        model = MLPModel(8, m['units'], m['layers'])
        model.train(dataset, m['lr'], m['batch_size'], m['epochs'], verbose=True, save_weights=False)


def train_model():
    """
    Train the model with the default architecture.
    """
    print('Loading dataset...')
    dataset = Dataset('training_set.csv')

    model = MLPModel(pre_trained_model=False)
    # model.train(dataset, lr=0.0005, batch_size=1000, nb_epoch=15, verbose=True, save_weights=True)
    model.train(dataset, lr=0.0010, batch_size=1000, nb_epoch=15, verbose=True, save_weights=True)


def predict():
    """
    Load latest pre-trained model and run a prediction
    """
    print('Loading dataset...')
    dataset = Dataset('training_set.csv', 'section_emplacement_gps.csv')

    model = MLPModel(pre_trained_model=True, verbose=True)

    # get today date string
    target_date = datetime.datetime.now()
    target_date_str = target_date.strftime("%y-%m-%d")
    print('Predicting occupancy rates for {}'.format(target_date_str))

    # get weather hourly temperature and day rain in mm
    with open('weather_forecast.json', 'r') as json_file:
        json_str = json_file.read()
        json_data = json.loads(json_str)
        print('day_precipitation_mm = {}'.format(json_data['day_precipitation_mm']))

    # generate the input matrix for inference
    (X, _) = dataset.get_input_matrix(target_date_str,
                                      hourly_temperature=json_data['hourly_temp'],
                                      day_rain_mm=json_data['day_precipitation_mm'])
    y_pred = model.predict(X)

    print(y_pred.shape)

    np.savetxt("X.csv", X, delimiter=",")
    np.savetxt("Y_pred.csv", y_pred, delimiter=",")


def test_predict():
    """
    Load latest pre-trained model and run a single prediction as testing.
    """
    model = MLPModel(pre_trained_model=True, verbose=True)

    # 2015-01-02 -- Friday(4) -- day of year 2 -- Section 1  -- temp -11 celcius -- rain 0.1mm -- occupancy ==> 0.1029
    # 2016-01-01 -- Friday(4) -- day of year 1 -- Section 1  -- temp  -2 celcius -- rain 1.0mm -- occupancy ==> 0.0843
    y_pred = model.predict_sample(day_of_year=2, day_of_week=4, hour=14,
                                  section_longitude=-73.570339, section_lattitude=45.507996,
                                  section_parking_count=17, temperature=-11, day_rain_mm=0.1)
    print(y_pred)

if __name__ == '__main__':
    """
    Arguments:
      --predict
      --train
      --experiments
      --test-predict
    """
    print('------------------------------------------')
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg == '--predict':
        predict()

    elif arg == '--train':
        train_model()

    elif arg == '--experiments':
        run_experiments()

    elif arg == '--test-predict':
        print('one sample test prediction')
        test_predict()

    else:
        print('Usage:')
        print('  --experiments')
        print('  --train')
        print('  --predict')
        print('  --test-predict')

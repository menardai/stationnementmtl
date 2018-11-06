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


from mlp_model import MLPModel
from dataset import Dataset


class Maps(object):
    """
    Multi-Layer Perceptron Model for predicting parking occupancy.
    """

    def __init__(self):
        pass

    def predict(self, dataset_filename, section_emplacement_gps_filename, weather_forecast_filename):
        """
        Load latest pre-trained model and run a prediction.
        Save input matrix 'X' and prediction list 'y' as instance variables.
        """
        print('Loading dataset...')
        self.dataset = Dataset(dataset_filename, section_emplacement_gps_filename)

        model = MLPModel(pre_trained_model=True, verbose=True)

        # get today date string
        target_date = datetime.datetime.now()
        target_date_str = target_date.strftime("%y-%m-%d")
        print('Predicting occupancy rates for {}'.format(target_date_str))

        # get weather hourly temperature and day rain in mm
        with open(weather_forecast_filename, 'r') as json_file:
            json_str = json_file.read()
            json_data = json.loads(json_str)
            print('day_precipitation_mm = {}'.format(json_data['day_precipitation_mm']))

        # generate the input matrix for inference
        (self.X, self.X_info_df) = self.dataset.get_input_matrix(target_date_str,
                                                                 hourly_temperature=json_data['hourly_temp'],
                                                                 day_rain_mm=json_data['day_precipitation_mm'])
        y_pred = model.predict(self.X)

        print('prediction one hot - shape {}'.format(y_pred.shape))
        self.y = [np.argmax(y, axis=None, out=None) for y in y_pred]
        # print('prediction from 0 to 4: {} values\n{}'.format(len(self.y), self.y))

    def save_occupancy_by_parking_spot(self, output_filename):
        # add occupancy prediction to input info data frame (section_id, hour).
        # 'df' data frame will have 3 columns: [hour, section_id, occupancy]
        df = self.X_info_df.copy(deep=True)
        df['occupancy'] = self.y

        # save a csv file per hour
        hour_list = df['hour'].unique()
        for hour in hour_list:
            print('saving file for hour {}'.format(hour))
            
            result = []
            occupancy_hour_df = df[df['hour'] == hour]

            # loop over every road section for this hour
            for (_, hour_row) in occupancy_hour_df.iterrows():
                section_parking_spot_df = self.dataset.parking_spot_gps(hour_row.section_id)

                # loop over every parking spot of this road section (for this hour)
                for (_, section_row) in section_parking_spot_df.iterrows():
                    # add a record for this parking spot.
                    # occupancy from the df [0, 4], adding one to get values [1, 5]
                    result.append([float(section_row.latitude), float(section_row.longitude), int(hour_row.occupancy + 1)])

            # serialize to JSON
            result_dict = {"occupancies": result}
            with open(output_filename.replace('HH', str(hour)), "w") as json_file:
                json.dump(result_dict, json_file)


if __name__ == '__main__':
    maps = Maps()
    maps.predict('training_set.csv', 'section_emplacement_gps.csv', 'weather_forecast.json')
    maps.save_occupancy_by_parking_spot('occupancy_google_maps_HH.json')

import numpy as np
import pandas as pd

import datetime
import json

from keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler

class Dataset(object):
    """
    Parse a parking cvs file from "Stationnement de Montreal".

    Instance Variables:
        X_train -- numpy array of shape (n, 8)

                   [day_of_year, day_of_week, hour,
                    section_longitude, section_lattitude, section_parking_count,
                    temperature, day_rain_mm]

        Y_train -- numpy array of shape (n, 1)

    """

    def __init__(self, dataset_filename, section_emplacement_gps_filename=None):
        """
        Parse the dataset csv file to extract X_train, Y_train, the feature scaling and the section list.

        Arguments:
            dataset_filename -- String, csv file of samples
                [ NoTroncon ; DateCreation ; DayOfYear ; DayOfWeek ; NoHeure ; TxOccupation ;
                  CenterLongitude ; CenterLatitude ; EmplacementCount ; Temp ; Precip_total_day_mm ]

            section_emplacement_gps_filename -- String, csv file of each spot with gps coordinates and it's corresponding section id
                [ NoTroncon ; CenterLongitude ; CenterLatitude ; EmplacementCount ; sNoEmplacement ;
                  longitude ; latitude ]
        """
        (self.X_train, self.Y_train, self.sc, self.section_list) = Dataset._parse_data_frame(dataset_filename, sep=';')

        if section_emplacement_gps_filename:
            self.section_spot_gps = pd.read_csv(section_emplacement_gps_filename, sep=';')

    def parking_spot_gps(self, section_id):
        """
        Returns longitude and latitude of every parking spot in the specified section.

        Arguments:
            section_id -- int, the section id of the section (troncon in French)

        Returns:
             spot_gps -- data frame, longitude and latitude of every parking spot in the specified section
        """
        section = self.section_spot_gps[self.section_spot_gps['NoTroncon'] == section_id]
        return section[['longitude', 'latitude']]

    def _parse_data_frame(filename, sep):
        """

        filename csv structure:
            Data columns (total 10 columns):
                NoTroncon           int64
                DateCreation        object
                DayOfYear           int64
                DayOfWeek           int64
                NoHeure             int64
                TxOccupation        float64
                CenterLongitude     float64
                CenterLatitude      float64
                EmplacementCount    int64
                Temp                float64
                Precip_total_day_mm float64

        Returns:
            X_train -- numpy array of shape (n, 8)
                [DayOfYear, DayOfWeek, NoHeure, CenterLongitude, CenterLatitude, EmplacementCount, Temperature, Precip_total_day_mm]

            Y_train -- numpy array of shape (n, 5)
                one hot encoded value, 5 categories. each

            sc -- feature scaling
        """
        # Importing the dataset
        dataset = pd.read_csv(filename, sep)

        # Training matrix Y
        # convert TxOccupation column from float to int [1,2,3,4,5]
        dataset.loc[(dataset.TxOccupation >= 0.80) & (dataset.TxOccupation <= 1.00),'TxOccupation'] = 4
        dataset.loc[(dataset.TxOccupation >= 0.60) & (dataset.TxOccupation < 0.80), 'TxOccupation'] = 3
        dataset.loc[(dataset.TxOccupation >= 0.40) & (dataset.TxOccupation < 0.60), 'TxOccupation'] = 2
        dataset.loc[(dataset.TxOccupation >= 0.20) & (dataset.TxOccupation < 0.40), 'TxOccupation'] = 1
        dataset.loc[(dataset.TxOccupation >= 0.00) & (dataset.TxOccupation < 0.20), 'TxOccupation'] = 0
        dataset.TxOccupation = dataset.TxOccupation.astype('int')

        y = dataset['TxOccupation'].values
        Y_train = to_categorical(y)

        # Training matrix X
        dataset.drop('TxOccupation', 1, inplace=True)
        X_train = dataset.iloc[:, 2:].values

        section_list = dataset['NoTroncon'].unique()

        # features scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)

        return (X_train, Y_train, sc, section_list)

    def get_input_matrix(self, date_str, day_temperature=8, day_rain_mm=0, hourly_temperature=None):
        """
        Create and return the model's input matrix for the specified date, temperature and rain.

        Arguments:
            date_str --
            hourly_temperature --
            day_temperature --
            day_rain_mm --

        Returns:
            input_matrix -- numpy array, shape (n, 8)
        """
        # extract day of year and day of week from date string
        date = datetime.date(*map(int, date_str.split('-')))
        day_of_year = date.timetuple().tm_yday
        day_of_week = date.timetuple().tm_wday

        rows_list = []
        rows_info_list = []

        for section_id in self.section_list:
            # get CenterLongitude, CenterLattitude, EmplacementCount for the given section id
            section_gps_row = self.section_spot_gps.loc[self.section_spot_gps['NoTroncon'] == section_id]

            hour_range = range(9, 22) if 0 <= day_of_week <= 4 else range(9, 18)
            for hour in hour_range:
                temperature = day_temperature if hourly_temperature == None else hourly_temperature[str(hour)]

                rows_list.append({
                    'DayOfYear': day_of_year,
                    'DayOfWeek': day_of_week,
                    'NoHeure': hour,
                    'CenterLongitude': section_gps_row.iat[0,1],
                    'CenterLatitude': section_gps_row.iat[0,2],
                    'EmplacementCount': section_gps_row.iat[0,3],
                    'Temp': temperature,
                    'Precip_total_day_mm': day_rain_mm
                })

                rows_info_list.append({
                    'section_id': section_id,
                    'hour': hour
                })

        # create a data frame from row list and reorder the columns
        input_matrix_df = pd.DataFrame(rows_list)
        input_matrix_df = input_matrix_df[['DayOfYear', 'DayOfWeek', 'NoHeure', 'CenterLongitude', 'CenterLatitude',
                                           'EmplacementCount', 'Temp', 'Precip_total_day_mm']]

        # create a data frame that match 'section id' and 'hour' with the input matrix rows
        input_info_df = pd.DataFrame(rows_info_list)

        return (input_matrix_df.values, input_info_df)

    def preprocessing(source_occupancy_filename, source_parking_spot_gps_filename, weather_filename):
        """
        Read source data csv file and process them to create the training set csv file.

        Arguments:
            source_occupancy_filename -- String, TxOccupation_QDS_2015et16.csv from Stationnement de Montreal
            source_parking_spot_gps_filename -- String, a subset of "Liste des places associees aux troncons du QDS.xlsx"
                                                from Stationnement de Montreal
            weather_filename -- String, hour temperature and day precipitation mm

        Returns:
            training_set.csv -- File, training set ready to be trained by the model
            section_emplacement_gps.csv -- File, parking spot gps with sections info
        """
        # --- Occupancy rate ---
        occupancy_df = pd.read_csv(source_occupancy_filename, sep=';', decimal=",")

        # Remove rows with 0 available minutes
        occupancy_df = occupancy_df[occupancy_df['NbMinDispo'] > 0]

        # remove some columns
        occupancy_df.drop('DescriptionTroncon', 1, inplace=True)
        occupancy_df.drop('NbMinDispo', 1, inplace=True)

        # convert NoHeure and NoTroncon columns from int to category
        occupancy_df.NoHeure = occupancy_df.NoHeure.astype('int')
        occupancy_df.NoTroncon = occupancy_df.NoTroncon.astype('int')

        # convert DateCreation from Object to Date
        occupancy_df.DateCreation = pd.to_datetime(occupancy_df['DateCreation'])

        # add columns day of week and days of year
        occupancy_df.insert(loc=2, column='DayOfYear', value=occupancy_df['DateCreation'].dt.dayofyear)
        occupancy_df.insert(loc=3, column='DayOfWeek', value=occupancy_df['DateCreation'].dt.dayofweek)

        # --- Parking spot GPS ---
        spot_gps_df = pd.read_csv(source_parking_spot_gps_filename, sep=',', decimal=".")
        section_gps_df  = pd.DataFrame(columns = ['NoTroncon', 'CenterLongitude', 'CenterLatitude', 'EmplacementCount'])

        spot_gps_df.rename(columns = {'nNoTroncon':'NoTroncon'}, inplace = True)

        for i, no_troncon in enumerate(spot_gps_df.NoTroncon.unique()):
            center_longitude = np.mean(spot_gps_df[spot_gps_df['NoTroncon'] == no_troncon]['longitude'])
            center_latitude = np.mean(spot_gps_df[spot_gps_df['NoTroncon'] == no_troncon]['latitude'])
            emplacement_count = len(spot_gps_df[spot_gps_df['NoTroncon'] == no_troncon])
            section_gps_df.loc[i] = [
                str(no_troncon),
                float("{0:.6f}".format(center_longitude)),
                float("{0:.6f}".format(center_latitude)),
                emplacement_count]

        section_gps_df.NoTroncon = section_gps_df.NoTroncon.astype('int')
        section_gps_df.EmplacementCount = section_gps_df.EmplacementCount.astype('int')

        # --- merge section with spot gps and save ---
        section_emplacement_gps = pd.merge(section_gps_df, spot_gps_df, on='NoTroncon')
        section_emplacement_gps.to_csv('section_emplacement_gps.csv', sep=';', index=False)

        # --- merge occupancy and section GPS ---
        occupancy_gps_df = pd.merge(occupancy_df, section_gps_df, on='NoTroncon')

        # --- Weather ---
        weather_df = pd.read_csv(weather_filename, sep=';', decimal=".")

        # remove and rename columns
        weather_df.drop('Temp_mercure', 1, inplace=True)
        weather_df.drop('Temp_eolien', 1, inplace=True)
        weather_df.rename(columns = {'Heure':'NoHeure'}, inplace = True)

        # convert DateCreation from Object to Date
        weather_df.rename(columns = {'Date':'DateCreation'}, inplace = True)
        weather_df.DateCreation = pd.to_datetime(weather_df['DateCreation'])

        # --- merge and save ---
        parking_weather_df = pd.merge(occupancy_gps_df, weather_df, how='left', left_on=['DateCreation', 'NoHeure'], right_on=['DateCreation', 'NoHeure'])
        parking_weather_df.to_csv('training_set.csv', sep=';', index=False)


if __name__ == '__main__':
    print('-----------------------------')
    print('Creating dataset...')

    Dataset.preprocessing('TxOccupation_QDS_2015et16.csv',
                          'emplacement_gps.csv',
                          'meteo_montreal.csv')

    print('Loading dataset...\n')
    dataset = Dataset('training_set.csv', 'section_emplacement_gps.csv')

    print('X_train shape: {}'.format(dataset.X_train.shape))
    print('Y_train shape: {}'.format(dataset.Y_train.shape))
    print()
    print('Standard Scaler mean:')
    print(dataset.sc.mean_)
    print()

    with open('weather_forecast.json', 'r') as json_file:
        json_str = json_file.read()
        json_data = json.loads(json_str)
        print('day_precipitation_mm = {}'.format(json_data['day_precipitation_mm']))

    (input, input_info_df) = dataset.get_input_matrix("2015-01-02",
                                                   hourly_temperature=json_data['hourly_temp'],
                                                   day_rain_mm=json_data['day_precipitation_mm'])
    # (input, input_info) = dataset.get_input_matrix("2015-01-02", day_temperature=-10, day_rain_mm=0.5)

    print('Input shape for 2015-01-02 = {} {}'.format(input.shape, len(input_info_df)))

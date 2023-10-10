import pandas as pd
import tensorflow as tf
import numpy as np


def casualty_to_one_hot(Y_labels):

    transf = {
        'Slight': 0,
        'Assistance': 1
    }

    Y_labels.replace(transf, inplace = True)

    return tf.one_hot(Y_labels, 2)

def one_hot_to_casualty(Y_labels):

    transf = {
        0: 'Slight',
        1: 'Assistance'
    }   

    return Y_labels.replace(transf)

def remove_features(data_frame):
    COLUMNS_TO_GET = ['AMG_X', 'AMG_Y', 'ROAD_TYPE', 'ACCIDENTTIME', 'NO_OF_VEHICLES',
                      'SURFACE_COND', 'SPEED_ZONE',
                      'LIGHT_CONDITION', 'ATMOSPH_COND', 'dia_semana', 'semana_en_año',
                      'VEHICLE_TYPE', 'VEHICLE_YEAR_MANUF', 'ACCIDENT_TYPE',
                      'ROAD_USER_TYPE', 'SEX', 'AGE',
                      'INJ_LEVEL',
                      'Lat', 'Long']

    data_frame = data_frame.loc[:, data_frame.columns.isin(COLUMNS_TO_GET)]
    data_frame = data_frame[COLUMNS_TO_GET]

    RENAMED_COLUMNS = ['Easting', 'Northing', '1st Road Class', 'Accident Time', 'Number of Vehicles',
                       'Road Surface', 'Speed Limit',
                       'Lighting Conditions', 'Weather Conditions', 'dia_semana', 'semana_en_año',
                       'Type of Vehicle', 'Age of Vehicle', 'First Point of Impact',
                       'Casualty Class', 'Sex of Casualty', 'Age of Casualty',
                       'Casualty Severity',
                       'latitude', 'longitude']


    data_frame.columns = RENAMED_COLUMNS
    
    return data_frame



def transform_hour_to_day_or_night(data_frame):
    data_frame['Accident Time'] = data_frame['Accident Time'].astype(int)
    accident_time = pd.DatetimeIndex(data_frame['Accident Time'])

    data_frame['Accident Time'] = data_frame['Accident Time'].mask(data_frame['Accident Time'] < 600, 2)
    data_frame['Accident Time'] = data_frame['Accident Time'].mask(data_frame['Accident Time'] > 1800, 2)
    data_frame['Accident Time'] = data_frame['Accident Time'].mask(data_frame['Accident Time'].between(600, 1800), 1)

    return data_frame

def accident_time_fake_to_hours_and_minutes(row):

    string_hours = int(row['Accident Time'][:2])
    string_minutes = int(row['Accident Time'][3::3])

    row['Accident Time Fake'] = row['Accident Time Fake'].replace(year = 2000, month = 1, day = 1, hour = string_hours, minute = string_minutes)

    return row['Accident Time Fake']

def accident_time_fake_to_seconds(row):

    row['Accident Time Fake'] = row['Accident Time Fake'].hour * 60 * 60 + row['Accident Time Fake'].minute * 60

    return row['Accident Time Fake']

def transform_hour_into_sin_cos(data_frame):
    data_frame['Accident Time Fake'] = pd.Series(pd.date_range("2000-01-01", periods=len(data_frame), freq="h"))

    data_frame['Accident Time Fake'] = data_frame.apply(lambda row: accident_time_fake_to_hours_and_minutes(row), axis = 1)
    data_frame['Accident Time Fake'] = data_frame.apply(lambda row: accident_time_fake_to_seconds(row), axis = 1)

    data_frame.rename({'Accident Time Fake': 'Second on Day'}, inplace = True,  axis='columns')

    seconds_in_day = 24*60*60

    data_frame['Accident Time Sin'] = np.sin(2*np.pi*data_frame['Second on Day']/seconds_in_day)
    data_frame['Accident Time Cos'] = np.cos(2*np.pi*data_frame['Second on Day']/seconds_in_day)

    data_frame.drop('Second on Day', axis=1, inplace=True)

    return data_frame

def clean_before_1(data_frame):
    target_class = 'Casualty Severity'

    AGE_OF_VEHICLE_VALUES_TO_REMOVE = ['XXXX']
    data_frame = data_frame[~data_frame['Age of Vehicle'].isin(AGE_OF_VEHICLE_VALUES_TO_REMOVE)]

    casualty_severity_replace = {
        '4': 'Slight', # Not injured
        4: 'Slight', # Not injured
        3: 'Assistance', # Other injury
        2: 'Assistance', # Serious injury
        1: 'Assistance' # Fatality
    }
    data_frame['Casualty Severity'].replace(casualty_severity_replace, inplace=True)

    # SEX_OF_CASUALTY
    SEX_OF_CASUALTY_VALUES_TO_REMOVE = ['U']
    sex_of_casualty_replace = {
        'M': 0,
        'F': 1
    }
    data_frame = data_frame[~data_frame['Sex of Casualty'].isin(SEX_OF_CASUALTY_VALUES_TO_REMOVE)]
    data_frame['Sex of Casualty'].replace(sex_of_casualty_replace, inplace=True)


    for i, road_class in enumerate(data_frame['1st Road Class'].value_counts().index):
        data_frame.loc[data_frame['1st Road Class'] == road_class, '1st Road Class'] = i


    data_frame['Age of Casualty'] = data_frame['Age of Casualty'].astype('int')
    data_frame['Age of Casualty'] = data_frame['Age of Casualty'].mask(data_frame['Age of Casualty'] < 18, 1)
    data_frame['Age of Casualty'] = data_frame['Age of Casualty'].mask(data_frame['Age of Casualty'].between(18, 25), 2)
    data_frame['Age of Casualty'] = data_frame['Age of Casualty'].mask(data_frame['Age of Casualty'].between(25, 65), 3)
    data_frame['Age of Casualty'] = data_frame['Age of Casualty'].mask(data_frame['Age of Casualty'] > 65, 4)

    # data_frame['Accident Time'] = data_frame['Accident Time'].str.replace(':', '')

    data_frame = transform_hour_into_sin_cos(data_frame)


    SEVERITY_TYPE_REPLACE = {1: 'Assistance',
                             2: 'Assistance',
                             3: 'Slight'
                            }

    data_frame[target_class].replace(SEVERITY_TYPE_REPLACE, inplace = True)

    data_frame['Weather Conditions'] = data_frame['Weather Conditions'].astype('int')
    data_frame['Casualty Class']     = data_frame['Casualty Class'].astype('int')
    data_frame['Age of Vehicle']     = data_frame['Age of Vehicle'].astype('int')
    data_frame['1st Road Class']     = data_frame['1st Road Class'].astype('int')
    data_frame['Sex of Casualty']    = data_frame['Sex of Casualty'].astype('int')


    data_frame['Speed Limit'] = data_frame['Speed Limit'].astype(int)

    data_frame = data_frame.drop_duplicates()
    data_frame = data_frame.dropna()
    data_frame = data_frame.reset_index(drop = True)

    return data_frame


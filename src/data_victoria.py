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
        'Not injured ': 'Slight',
        'Other injury': 'Assistance',
        'Serious injury': 'Assistance',
        'Fatality': 'Assistance'
    }
    data_frame['Casualty Severity'].replace(casualty_severity_replace, inplace=True)

    # SEX_OF_CASUALTY
    SEX_OF_CASUALTY_VALUES_TO_REMOVE = ['Unknown']
    sex_of_casualty_replace = {
        'Male': 0,
        'Female': 1
    }
    data_frame = data_frame[~data_frame['Sex of Casualty'].isin(SEX_OF_CASUALTY_VALUES_TO_REMOVE)]
    data_frame['Sex of Casualty'].replace(sex_of_casualty_replace, inplace=True)

    # AGE_OF_CASUALTY
    AGE_OF_CASUALTY_VALUES_TO_REMOVE = ['XX', 'XXX']
    data_frame = data_frame[~data_frame['Age of Casualty'].isin(AGE_OF_CASUALTY_VALUES_TO_REMOVE)]

    # WEATHER CONDITIONS
    WEATHER_CONDITIONS_VALUES_TO_REMOVE = ['Unknown']
    weather_conditions_replace = {
        'Not Raining': 0,
        'Raining': 1
    }
    data_frame = data_frame[~data_frame['Weather Conditions'].isin(WEATHER_CONDITIONS_VALUES_TO_REMOVE)]
    data_frame['Weather Conditions'].replace(weather_conditions_replace, inplace=True)


    # LIGHT_CONDITIONS
    light_conditions_replace = {'Daylight': 0,
                                'Night': 1}
    data_frame['Lighting Conditions'].replace(light_conditions_replace, inplace = True)


    # FIRST_POINT_OF_IMPACT
    first_point_of_impact_replace = {
        'Hit Pedestrian': 0,
        'Roll Over': 1,
        'Hit Object on Road': 2,
        'Left Road - Out of Control': 3,
        'Head On': 4,
        'Rear End': 5,
        'Right Angle': 6,
        'Side Swipe': 7, 
        'Right Turn': 8,
        'Hit Fixed Object': 9,
        'Hit Animal': 10,
        'Hit Parked Vehicle': 11,
        'Other': 12
    }
    data_frame['First Point of Impact'].replace(first_point_of_impact_replace, inplace = True)

    # WEATHER CONDITIONS
    ROAD_SURFACE_CONDITIONS_VALUES_TO_REMOVE = ['Unknown']
    weather_conditions_replace = {
        'Sealed': 0,
        'Unsealed': 1
    }
    data_frame = data_frame[~data_frame['Road Surface'].isin(ROAD_SURFACE_CONDITIONS_VALUES_TO_REMOVE)]
    data_frame['Road Surface'].replace(weather_conditions_replace, inplace=True)


    type_of_vehicle_replace = {
        'Scooter': 0,
        'Motor Cycle': 1,
        'Motor Cars - Sedan': 2,
        'Motor Cars - Tourer': 3,
        'Taxi Cab': 4,
        'Station Wagon': 5,
        'Utility': 6,
        'Motor Vehicle - Type Unknown': 6,
        'Panel Van': 7,
        'Forward Control Passenger Van': 7,
        'OMNIBUS': 8,
        'SEMI TRAILER': 9,
        'Light Truck LT 4.5T': 10,
        'RIGID TRUCK LGE GE 4.5T': 11,
        'BDOUBLE - ROAD TRAIN': 12,
        'Other Defined Special Vehicle': 13
    }

    data_frame['Type of Vehicle'].replace(type_of_vehicle_replace, inplace=True)

    first_road_class_replace = {
        'Freeway': 0,
        'Not Divided': 1,
        'One Way': 2,
        'Divided Road': 3,
        'Multiple': 4,
        'Interchange': 5,
        'Cross Road': 6,
        'Crossover': 7,
        'T-Junction': 8,
        'Y-Junction': 9,
        'Ramp On': 10,
        'Ramp Off': 11,
        'Rail Xing': 12,
        'Rail Crossing': 13,
        'Pedestrian Crossing': 14,
        'Other': 10
    }
    data_frame['1st Road Class'].replace(first_road_class_replace, inplace = True)


    casualty_class_replace = {
        'Driver': 0,
        'Passenger': 1,
        'Rider': 2,
    }
    data_frame['Casualty Class'].replace(casualty_class_replace, inplace = True)


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

    data_frame['Speed Limit'] = data_frame['Speed Limit'].astype(int)

    data_frame = data_frame.drop_duplicates()
    data_frame = data_frame.dropna()
    data_frame = data_frame.reset_index(drop = True)

    return data_frame


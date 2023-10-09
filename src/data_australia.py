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
    COLUMNS_TO_GET = ['ACCLOC_X', 'ACCLOC_Y', 'Position Type', 'Crash Date Time', 'Total Units',
                      'Road Surface', 'Area Speed',
                      'DayNight', 'Weather Cond', 'dia_semana', 'semana_en_año',
                      'Unit Type', 'Veh Year', 'Crash Type',
                      'Casualty Type', 'Sex of Casualty', 'Age of Casualty',
                      'Casualty Severity',
                      'longitude', 'latitude']

    data_frame = data_frame.loc[:, data_frame.columns.isin(COLUMNS_TO_GET)]
    data_frame = data_frame[COLUMNS_TO_GET]

    RENAMED_COLUMNS = ['Easting', 'Northing', '1st Road Class', 'Accident Time', 'Number of Vehicles',
                       'Road Surface', 'Speed Limit',
                       'Lighting Conditions', 'Weather Conditions', 'dia_semana', 'semana_en_año',
                       'Type of Vehicle', 'Age of Vehicle', 'First Point of Impact',
                       'Casualty Class', 'Sex of Casualty', 'Age of Casualty',
                       'Casualty Severity',
                       'longitude', 'latitude']


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

    string_hours = row['Accident Time'].hour
    string_minutes = row['Accident Time'].minute

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

    ROAD_SURFACE_VALUES_TO_REMOVE = [-1, 6, 7, 9]

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

    data_frame['Type of Vehicle'] = data_frame['Type of Vehicle'] + 1
    data_frame.loc[data_frame['Type of Vehicle'] > 10, 'Type of Vehicle'] -= 1
    data_frame.loc[data_frame['Type of Vehicle'] > 11, 'Type of Vehicle'] -= 1

    data_frame = data_frame[data_frame['First Point of Impact'] != -1]
    data_frame['First Point of Impact'] = data_frame['First Point of Impact'] + 1


    data_frame['Type of Vehicle'] = data_frame['Type of Vehicle'] + 1

    data_frame = data_frame[data_frame['Sex of Casualty'] != -1]

    data_frame = data_frame[data_frame['Sex of Casualty'] != -1]

    data_frame = data_frame[data_frame['Age of Vehicle'] != -1]

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

    data_frame['Northing'] = data_frame['Northing'].astype(int)
    data_frame['Easting']  = data_frame['Easting'].astype(int)
    data_frame['Speed Limit'] = data_frame['Speed Limit'].astype(int)

    data_frame = data_frame.drop_duplicates()
    data_frame = data_frame.dropna()
    data_frame = data_frame.reset_index(drop = True)

    return data_frame



def clean_before_2(data_frame):
    ###################### DICCIONARIOS DE REEMPLAZO ######################
    # Unclassified: Carreteras locales sin destino definido. Sin embargo, los destinos locales pueden estar señalizados a lo largo de ellos.
    # A, A(M) y Motorway lo mismo?
    # B:            De carácter regional y utilizado para conectar zonas de menor importancia.
    #               Por lo general, se muestran de color marrón o amarillo en los mapas y tienen las mismas señales blancas que las rutas de clase A que no son primarias.
    #               Si la ruta es primaria, como la B6261, se mostrará igual que una ruta Clase A primaria.
    #               ¿Carretera como tal?

    # C:            Designaciones de autoridades locales para rutas dentro de su área con fines administrativos.
    #               Estas rutas no se muestran en mapas de carreteras a pequeña escala, pero se sabe que ocasionalmente aparecen en las señales de tráfico.
    road_class_replace = {
        'Motorway': 1,
        'A(M)': 2,
        'A': 3,
        'B': 4,
        'C': 5,
        'Unclassified': 6
    }

    accident_date_replace = {
        'Dry': 1,
        'Wet / Damp': 2,
        'Snow': 3,
        'Frost / Ice': 4,
        'Flood': 5,
    }

    ##################################
    road_surface_replace = {
        '1': 1, # Dry
        '2': 2, # Wet
        '3': 3, # Snow
        '4': 4, # Snow
        '5': 5, # Icy
        '6': 6, # Sand/gravel/dirt
        '8': 8, # Oil
        '9': 9, # Flooded
        'Q': -1,
        'U': -1,
        'X': -1
    }

    
    dataset['Road Surface'].replace(road_surface_replace, inplace=True)

    # La 5: "Darkness: street lighting unknown" no está presente en el paper, le hemos puesto un 5 porque sí #
    lighting_conditions_replace = {
        'Daylight: street lights present': 1,
        'Darkness: no street lighting': 2,
        'Darkness: street lights present and lit': 3,
        'Darkness: street lights present but unlit': 4,
        'Darkness: street lighting unknown': 5,
        '5': 5
    }

    ##################################
    weather_conditions_replace = {
        '1': 1, # Clear and sunny
        '2': 2, # Overcast, cloudy but no precipitation
        '3': 3, # Raining
        '4': 4, # Snowing, not including drifting snow
        '5': 5, # Freezing rain, sleet, hail
        '6': 6, # Visibility limitation
        '7': 7, # Strong wind
        'Q': -1,
        'U': -1,
        'X': -1
    }

    dataset['Weather Conditions'].replace(weather_conditions_replace, inplace=True)




    type_of_vehicle_replace = {
        'Pedal cycle': 1,
        'M/cycle 50cc and under': 2,
        'Motorcycle over 50cc and up to 125cc': 3,
        'Motorcycle over 125cc and up to 500cc': 4,
        'Motorcycle over 500cc': 5,
        'Taxi/Private hire car': 6,
        'Car': 7,
        'Minibus (8 – 16 passenger seats)': 8,
        'Bus or coach (17 or more passenger seats)': 9,
        'Ridden horse': 10,
        'Agricultural vehicle (includes diggers etc.)': 11,
        'Tram / Light rail': 12,
        'Goods vehicle 3.5 tonnes mgw and under': 13,
        'Goods vehicle over 3.5 tonnes and under 7.5 tonnes mgw': 14,
        'Goods vehicle 7.5 tonnes mgw and over': 15,
        'Mobility Scooter': 16,
        'Other Vehicle ': 17,
        'Motorcycle - Unknown CC': 18
    }
    casualty_class_replace = {
        'Driver': 1,
        'Driver/Rider': 1,
        'Driver or rider': 1,
        'Passenger': 2,
        'Vehicle or pillion passenger': 2,
        'Pedestrian': 3
    }


    sex_of_casualty_replace = {
        'Male': 1,
        'Female': 2
    }

    a = clean_df = data_frame

    ###################### REEMPLAZOS ######################
    clean_df = clean_df.dropna()

    a['1st Road Class'].replace(road_class_replace, inplace = True)
    # print('1st Road Class:', a['1st Road Class'].unique())

    ##################################
    # a['Accident Date'].replace(accident_date_replace, inplace = True)
    # print('Accident Date:', a['Accident Date'].unique())
    ##################################
    a['Road Surface'].replace(road_surface_replace, inplace = True)
    a.dropna(inplace = True)

    a['Road Surface'] = a['Road Surface'].astype('int')
    # print('Road Surface:', a['Road Surface'].unique())

    a['Lighting Conditions'].replace(lighting_conditions_replace, inplace = True)
    # print('Lighting Conditions:', a['Lighting Conditions'].unique())

    a['Weather Conditions'].replace(weather_conditions_replace, inplace = True)
    a = a[a['Weather Conditions'] != 'Darkness: street lighting unknown']
    # print('Weather Conditions:', a['Weather Conditions'].unique())

    a['Type of Vehicle'].replace(type_of_vehicle_replace, inplace = True)
    # print('Type of Vehicle:', a['Type of Vehicle'].unique())

    a['Casualty Class'].replace(casualty_class_replace, inplace = True)
    # print('Casualty Class:', a['Casualty Class'].unique())

    a['Sex of Casualty'].replace(sex_of_casualty_replace, inplace = True)
    # print('Sex of Casualty:', a['Sex of Casualty'].unique())

    a['Age of Casualty'] = a['Age of Casualty'].mask(a['Age of Casualty'] < 18, 1)
    a['Age of Casualty'] = a['Age of Casualty'].mask(a['Age of Casualty'].between(18, 25), 2)
    a['Age of Casualty'] = a['Age of Casualty'].mask(a['Age of Casualty'].between(25, 65), 3)
    a['Age of Casualty'] = a['Age of Casualty'].mask(a['Age of Casualty'] > 65, 4)
    # print('Age of Casualty:', a['Age of Casualty'].unique())

    a['Accident Time'] = a['Accident Time'].str.replace(':', '')
    a['Accident Time'] = a['Accident Time'].astype(int)

    a['Accident Time'] = a['Accident Time'].mask(a['Accident Time'] < 600, 2)
    a['Accident Time'] = a['Accident Time'].mask(a['Accident Time'] > 1800, 2)
    a['Accident Time'] = a['Accident Time'].mask(a['Accident Time'].between(600, 1800), 1)
    # print('Time (24hr):', a['Time (24hr)'].unique())
    a['Accident Time'].astype(int)

    ###################### LIMPIEZA DE VALORES NULOS/DUPLICADOS ######################

    clean_df = a.loc[:, ~a.columns.isin(['Accident Date', 'Reference Number'])]

    clean_df['Weather Conditions'] = clean_df['Weather Conditions'].astype('int')
    clean_df['Casualty Class']     = clean_df['Casualty Class'].astype('int')

    clean_df = clean_df.drop_duplicates()
    clean_df = clean_df.dropna()
    clean_df = clean_df.reset_index(drop=True)

    return clean_df
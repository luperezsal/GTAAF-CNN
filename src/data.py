import pandas as pd
import re

def read_madrid_data(root_path):

    file_name_2019 = '2019_Accidentalidad.csv'
    file_name_2020 = '2020_Accidentalidad.csv'
    file_name_2021 = '2021_Accidentalidad.csv'
    file_name_2022 = '2022_Accidentalidad.csv'

    file_2019 = pd.read_csv(root_path + file_name_2019, sep=';')
    file_2020 = pd.read_csv(root_path + file_name_2020, sep=';')
    file_2021 = pd.read_csv(root_path + file_name_2021, sep=';')
    file_2022 = pd.read_csv(root_path + file_name_2022, sep=';')

    # print(len(file_2019[file_2019.cod_lesividad == 4]))
    # print(len(file_2020[file_2020.cod_lesividad == 4]))
    # print(len(file_2021[file_2021.lesividad == '4']))
    # print(len(file_2022[file_2022.lesividad == '4']))

    COLUMNS_TO_REMOVE = ['cod_distrito',
                         'tipo_lesividad'
                        ]

    data_frame = file_2019
    data_frame = pd.concat([data_frame, file_2020])

    data_frame.rename(columns={"cod_lesividad": "lesividad"}, inplace = True)
    data_frame.rename(columns={"tipo_vehículo": "tipo_vehiculo"}, inplace = True)
    data_frame = data_frame.drop(COLUMNS_TO_REMOVE, axis=1)

    data_frame = pd.concat([data_frame, file_2021])

    data_frame.dropna(subset=['lesividad'], inplace = True)
    data_frame.lesividad = data_frame.lesividad.replace(' ', 14).astype(int)
    data_frame = data_frame.reset_index(drop=True)
    
    return data_frame
    
def get_to_iso_calendar(data_frame_row):
    row_datetime = pd.to_datetime(data_frame_row['fecha'], format="%d/%m/%Y")

    return row_datetime.isocalendar()

######################################################

def categorize_features(data_frame) :
    weather_conditions_replace = {
        'Despejado': 1,
        'Nublado': 2,
        'Lluvia débil': 3,
        'LLuvia intensa': 4,
        'Granizando':  5,
        'Nevando': 6,
        'Se desconoce': 7 
    }

    ## CUIDADO CON Motocicleta hasta 125cc!!! HEMOS SUPUESTO QUE LOS CICLOMOTORES SON HASTA 50CC!!
    type_of_vehicle_replace = {
        'Bicicleta': 1,
        'Ciclo': 1,
        'Bicicleta EPAC (pedaleo asistido)': 1,
        'Ciclomotor': 2,
        'Ciclomotor de dos ruedas L1e-B': 2,
        'Ciclomotor de tres ruedas': 2,
        'Motocicleta hasta 125cc': 3,
        'Moto de tres ruedas hasta 125cc': 3,
        'Motocicleta > 125cc': 4,
        'Moto de tres ruedas > 125cc': 4,
        'Turismo': 5,
        'Todo terreno': 5,
        'Microbús <= 17 plazas': 5,
        'Autobús': 6,
        'Autobus EMT': 6,
        'Autobús articulado': 6,
        'Autobús articulado EMT': 6,
        'Maquinaria agrícola': 7,
        'Maquinaria de obras': 8,
        'Furgoneta': 9,        # Menos de 3.5 toneladas.
        'Ambulancia SAMUR': 10,
        'Autocaravana': 11,     # Entre 3.5 y 7.5 toneladas.
        'Camión rígido': 12,    # Mayor que 7.5 toneladas.
        'Tractocamión': 12,
        'Vehículo articulado': 12,
        'Camión de bomberos': 12,
        'VMU eléctrico': 13,
        'Patinete': 13,
        'Sin especificar': 14,
        'Otros vehículos sin motor': 14,
        'Remolque': 14,
        'Semiremolque': 14,
        'Otros vehículos con motor': 15,
        'Cuadriciclo ligero': 15,
        'Cuadriciclo no ligero': 15,
        'Motorcycle - Unknown CC': 15
    }

    # type_of_vehicle_replace = {}
    # for index,tipo_vehiculo in enumerate(data_frame.tipo_vehiculo.unique()):
    #     if not pd.isna(tipo_vehiculo): type_of_vehicle_replace[tipo_vehiculo] = index

    casualty_class_replace = {
        'Conductor': 1,
        'Pasajero': 2,
        'Peatón': 3
    }

    ### CUIDADO CON DESCONOCIDO!!! MEJOR HACER IMPUTACIÓN PARA RELLENENAR LOS DESCONOCIDOS?
    sex_of_casualty_replace = {
        'Hombre': 1,
        'Mujer': 2,
        'Desconocido': 3
    }

    accident_type_replace = {
        'Colisión fronto-lateral': 1,
        'Alcance': 2,
        'Colisión lateral': 3,
        'Choque contra obstáculo fijo': 4,
        'Colisión múltiple': 5,
        'Caída': 5,
        'Atropello a persona': 7,
        'Colisión frontal': 8,
        'Otro': 9,
        'Solo salida de la vía': 10,
        'Vuelco': 11,
        'Atropello a animal': 12,
        'Despeñamiento': 13
    }

    alcohol_replace = {
        'S': 1,
        'N': 2,
    }

    accident_class_replace = {
        1:  'Slight',  # Atención en urgencias sin posterior ingreso. - LEVE
        2:  'Slight',  # Ingreso inferior o igual a 24 horas - LEVE
        5:  'Slight',  # Asistencia sanitaria ambulatoria con posterioridad - LEVE
        6:  'Slight',  # Asistencia sanitaria inmediata en centro de salud o mutua - LEVE
        7:  'Slight',  # Asistencia sanitaria sólo en el lugar del accidente - LEVE
        14: 'Slight',  # Sin asistencia sanitaria - LEVE O NADA
        3:  'Serious', # Ingreso superior a 24 horas. - GRAVE
        4:  'Fatal'    # Fallecido 24 horas - FALLECIDO 
    }
    ###################### REEMPLAZOS ######################

    # ### OJO QUE ESTAMOS REPLICANDO LA ESTRUCTURA DEL DATASET DE LEEDS
    age_replace = {
        'Menor de 5 años': 1,
        'De 6 a 9 años': 1,
        'De 6  a  9 años': 1,
        'De 10 a 14 años': 1,
        'De 15 a 17 años': 1,
        'De 18 a 20 años': 2,
        'De 21 a 24 años': 2,
        'De 25 a 29 años': 3,
        'De 30 a 34 años': 3,
        'De 35 a 39 años': 3,
        'De 40 a 44 años': 3,
        'De 45 a 49 años': 3,
        'De 50 a 54 años': 3,
        'De 55 a 59 años': 3,
        'De 60 a 64 años': 3,
        'De 65 a 69 años': 4,
        'De 70 a 74 años': 4,
        'Más de 74 años': 4,
        'Desconocido': 5,
    }

    # age_replace = {
    #     'Menor de 5 años': 1,
    #     'De 6 a 9 años': 2,
    #     'De 6  a  9 años': 3,
    #     'De 10 a 14 años': 4,
    #     'De 15 a 17 años': 5,
    #     'De 18 a 20 años': 6,
    #     'De 21 a 24 años': 7,
    #     'De 25 a 29 años': 8,
    #     'De 30 a 34 años': 9,
    #     'De 35 a 39 años': 10,
    #     'De 40 a 44 años': 11,
    #     'De 45 a 49 años': 12,
    #     'De 50 a 54 años': 13,
    #     'De 55 a 59 años': 14,
    #     'De 60 a 64 años': 15,
    #     'De 65 a 69 años': 16,
    #     'De 70 a 74 años': 17,
    #     'Más de 74 años': 18,
    #     'Desconocido': 19,
    # }

    data_frame['estado_meteorológico'].replace(weather_conditions_replace, inplace = True)
    # print('Estado meteorológico: \n', data_frame['estado_meteorológico'].value_counts())

    data_frame['tipo_vehiculo'].replace(type_of_vehicle_replace, inplace = True)
    # print('Tipo vehículo: \n', data_frame['tipo_vehiculo'].value_counts())

    data_frame['tipo_persona'].replace(casualty_class_replace, inplace = True)
    # print('Tipo de persona: \n', data_frame['tipo_persona'].value_counts())

    data_frame['sexo'].replace(sex_of_casualty_replace, inplace = True)
    # print('Sexo: \n', data_frame['sexo'].value_counts())

    indexes_of_positive_drug = data_frame[data_frame.positiva_droga == 1].index
    data_frame.loc[indexes_of_positive_drug, 'positiva_alcohol'] = 'S'

    data_frame['positiva_alcohol'].replace(alcohol_replace, inplace = True)
    # print('Positivo Alcohol: \n', data_frame['positiva_alcohol'].value_counts())

    data_frame['lesividad'].replace(accident_class_replace, inplace = True)
    # print('Gravedad: \n', data_frame['lesividad'].value_counts())

    data_frame['rango_edad'].replace(age_replace, inplace = True)
    # print('Edad: \n', data_frame['rango_edad'].value_counts())

    data_frame.hora = data_frame.hora.mask(pd.to_datetime(data_frame.hora) < '06:00:00', 2)
    data_frame.hora = data_frame.hora.mask(pd.to_datetime(data_frame.hora) > '18:00:00', 2)
    data_frame.hora = data_frame.hora.mask(pd.to_datetime(data_frame.hora).between('06:00:00', '18:00:00'), 1)
    # print('hora:', data_frame['hora'].value_counts())

    district_replace = {}
    for index,distrito in enumerate(data_frame.distrito.unique()):
      if not pd.isna(distrito): district_replace[distrito] = int(index)

    accident_type_replace = {}
    for index,accident_type in enumerate(data_frame.tipo_accidente.unique()):
        if not pd.isna(accident_type): accident_type_replace[accident_type] = int(index)

    data_frame['distrito'].replace(district_replace, inplace = True)
    # print('Distrito: \n', data_frame['distrito'].value_counts())

    data_frame['tipo_accidente'].replace(accident_type_replace, inplace = True)
    # print('Tipo Accidente: \n', data_frame['tipo_accidente'].value_counts())

    # Eliminamos aquellas lesividades desconocidas i.e. 77.
    data_frame = data_frame[data_frame.lesividad != 77]

    return data_frame

######################################################

def utm_to_int_old(data_frame):
    # Todos las comas a puntos

    s = data_frame.coordenada_x_utm.str
    s_y = data_frame.coordenada_y_utm.str

    # Regex que hace match para dos grupos, la parte entera y la parte decimal.
    group_integer_and_float_pattern = '(?P<Integer>\d{3}\.\d{3})(?P<Float>\.\d{2,3})'
    all_float_pattern   = '(?P<Number>\d{6},\d+)'
    all_integer_pattern = '(?P<Number>\d{6}$)'

    group_integer_and_float_pattern_y = '(?P<Integer>\d\.\d{3}\.\d{3})(?P<Float>\.\d{2,3})'
    all_float_pattern_y   = '(?P<Number>\d{7},\d+)'
    all_integer_pattern_y = '(?P<Number>\d{7}$)'

    # Se extraen en un dataframe independiente ambas partes, la entera y la decimal
    index_and_extracted_x1 = s.extract(group_integer_and_float_pattern)
    index_and_extracted_x2 = s.extract(all_float_pattern)
    index_and_extracted_x3 = s.extract(all_integer_pattern)

    index_and_extracted_y1 = s_y.extract(group_integer_and_float_pattern_y)
    index_and_extracted_y2 = s_y.extract(all_float_pattern_y)
    index_and_extracted_y3 = s_y.extract(all_integer_pattern_y)

    # Se seleccionan aquellas que no continenen valores nulos el Float.
    # Es decir, aquellos con los que el match ha tenido éxito (los que llevan punto)
    # en lugar de comas.
    selected_rows_x1 = index_and_extracted_x1[~index_and_extracted_x1['Float'].isnull()]
    selected_rows_x2 = index_and_extracted_x2[~index_and_extracted_x2['Number'].isnull()]
    selected_rows_x3 = index_and_extracted_x3[~index_and_extracted_x3['Number'].isnull()]

    selected_rows_y1 = index_and_extracted_y1[~index_and_extracted_y1['Float'].isnull()]
    selected_rows_y2 = index_and_extracted_y2[~index_and_extracted_y2['Number'].isnull()]
    selected_rows_y3 = index_and_extracted_y3[~index_and_extracted_y3['Number'].isnull()]

    # Se cambia el string de la parte entera a un string sin puntos.
    selected_rows_x1.Integer = selected_rows_x1.Integer.str.replace('.','')
    selected_rows_x2.Number  = selected_rows_x2.Number.str.replace(',','.')

    selected_rows_y1.Integer = selected_rows_y1.Integer.str.replace('.','')
    selected_rows_y2.Number  = selected_rows_y2.Number.str.replace(',','.')

    # Se crea una nueva columna en el nuevo dataframe con la unión de la parte
    # entera y la parte decimal.
    selected_rows_x1['processed_x_utm'] = selected_rows_x1.Integer + selected_rows_x1.Float
    selected_rows_x2['processed_x_utm'] = selected_rows_x2.Number
    selected_rows_x3['processed_x_utm'] = selected_rows_x3.Number

    selected_rows_y1['processed_y_utm'] = selected_rows_y1.Integer + selected_rows_y1.Float
    selected_rows_y2['processed_y_utm'] = selected_rows_y2.Number
    selected_rows_y3['processed_y_utm'] = selected_rows_y3.Number

    data_frame['processed_x_utm'] = 'N/A'
    data_frame['processed_y_utm'] = 'N/A'

    # Si la longitud de alguno de los números es menor a diez, hay que añadirle x 0s
    # de diferencia
    selected_rows_x2.processed_x_utm = selected_rows_x2.processed_x_utm.transform(lambda x: x + '0'*(10-len(x)))
    selected_rows_x3.processed_x_utm = selected_rows_x3.processed_x_utm.transform(lambda x: x + '.000')

    selected_rows_y2.processed_y_utm = selected_rows_y2.processed_y_utm.transform(lambda x: x + '0'*(11-len(x)))
    selected_rows_y3.processed_y_utm = selected_rows_y3.processed_y_utm.transform(lambda x: x + '.000')

    data_frame['processed_x_utm'][selected_rows_x1.index] = selected_rows_x1['processed_x_utm']
    data_frame['processed_x_utm'][selected_rows_x2.index] = selected_rows_x2['processed_x_utm']
    data_frame['processed_x_utm'][selected_rows_x3.index] = selected_rows_x3['processed_x_utm']

    data_frame['processed_y_utm'][selected_rows_y1.index] = selected_rows_y1['processed_y_utm']
    data_frame['processed_y_utm'][selected_rows_y2.index] = selected_rows_y2['processed_y_utm']
    data_frame['processed_y_utm'][selected_rows_y3.index] = selected_rows_y3['processed_y_utm']

    # Eliminamos aquellas filas que no tienen coordenadas
    data_frame = data_frame[data_frame['coordenada_y_utm'] != '0.000']

    # Eliminamos el punto de la parte decimal para convertirlo a entero
    data_frame.processed_x_utm = data_frame.processed_x_utm.str.replace('.','')
    data_frame.processed_y_utm = data_frame.processed_y_utm.str.replace('.','')

    # Lo convertimos en entero
    data_frame.processed_x_utm = data_frame.processed_x_utm.astype(int)
    data_frame.processed_y_utm = data_frame.processed_y_utm.astype(int)

    return data_frame


import utm


def make_dirty_utm_to_clean_float_utm(utm_coordinate, int_part_number_of_digits):
    try:
        all_string_digit_list = re.findall(r"\d", utm_coordinate)
    except:
        print(utm_coordinate)
        return 0

    int_utm_part   = all_string_digit_list[:int_part_number_of_digits]
    float_utm_part = all_string_digit_list[int_part_number_of_digits:]

    int_utm_part   = ''.join(int_utm_part)
    float_utm_part = ''.join(float_utm_part)

    utm_coordinate = f"{int_utm_part}.{float_utm_part}"
    float_utm_coordinate = float(utm_coordinate)

    return float_utm_coordinate

def utm_to_int(data_frame):
    # Todos las comas a puntos


    int_part_number_of_digits = 6
    data_frame.coordenada_x_utm = data_frame.coordenada_x_utm.apply(lambda utm_coordinate: make_dirty_utm_to_clean_float_utm(utm_coordinate, int_part_number_of_digits))

    int_part_number_of_digits = 7
    data_frame.coordenada_y_utm = data_frame.coordenada_y_utm.apply(lambda utm_coordinate: make_dirty_utm_to_clean_float_utm(utm_coordinate, int_part_number_of_digits))

    data_frame = data_frame[data_frame.coordenada_y_utm != 0]
    data_frame = data_frame[data_frame.coordenada_x_utm != 0]

    data_frame = data_frame.assign(Latitude = lambda row: (utm.to_latlon(row.coordenada_x_utm, row.coordenada_y_utm, 30, 'T')[0]))
    data_frame = data_frame.assign(Longitude = lambda row: (utm.to_latlon(row.coordenada_x_utm, row.coordenada_y_utm, 30, 'T')[1]))

    return data_frame


######################################################
def remove_features(data_frame):
    COLUMNS_TO_REMOVE = ['num_expediente', 'fecha', 'tipo_via', 'numero', 'positiva_droga']

    data_frame = data_frame.loc[:, ~data_frame.columns.isin(COLUMNS_TO_REMOVE)]

    data_frame.rename(columns={"localizacion": "tipo_carretera"}, errors="raise", inplace=True)
    data_frame.rename(columns={"positiva_alcohol": "drogas_alcohol_positivo"}, errors="raise", inplace=True)

    data_frame = data_frame.drop_duplicates()
    data_frame = data_frame.dropna()
    data_frame = data_frame.reset_index(drop=True)
    
    return data_frame
######################################################

from tqdm import tqdm
import math


def get_intervals(data_frame, x_name='coordenada_x_utm', y_name='coordenada_y_utm'):

    min_x = int(data_frame[x_name].min())
    max_x = math.ceil(data_frame[x_name].max())

    min_y = int(data_frame[y_name].min())
    max_y = math.ceil(data_frame[y_name].max())

    interval_x = (max_x - min_x)
    interval_y = (max_y - min_y)

    return interval_x, interval_y


def get_divisible_numbers(number):

    for i in range (1, number):
        zero = number%i
        if zero == 0: print(f"Area units: {i}, regions: {number/i}")


def get_rows_by_removing_areas(data_frame, x_name='coordenada_x_utm', y_name='coordenada_y_utm', x_offset = 14311, y_offset = 17335, casualty_name='lesividad', casualty_target_names=['Fatal', 'Serious']):
    # Default: Madrid names

    ######################################################
    
    min_x = int(data_frame[x_name].min())
    max_x = math.ceil(data_frame[x_name].max())

    min_y = int(data_frame[y_name].min())
    max_y = math.ceil(data_frame[y_name].max())

    interval_x = (max_x - min_x)
    interval_y = (max_y - min_y)

    ######################################################

    # Se ejecuta 2 veces
    # for two_executions in range(1):
        # data_frame = data_frame[data_frame.coordenada_x_utm != min_x]
        # data_frame = data_frame[data_frame.coordenada_y_utm != min_y]

        # min_x = data_frame.coordenada_x_utm.min()
        # max_x = data_frame.coordenada_x_utm.max()

        # min_y = data_frame.coordenada_y_utm.min()
        # max_y = data_frame.coordenada_y_utm.max()

        # interval_x = (math.ceil(max_x) - int(min_x))
        # interval_y = (math.ceil(max_y) - int(min_y))

        
        
    ######################################################
 
    # def get_divisible_numbers(number):
    #     for i in range (1, number):
    #         zero = number%i
    #         if zero == 0: print(f"Divisible by: {i}, regions: {number/i}")

    # # Number: 1831.0, divisible number: 14311
    # # get_divisible_numbers(interval_x)
    # # Number of regions: 1810.0, divisible number: 17335
    # # get_divisible_numbers(interval_y)

    ######################################################

    initial_x = min_x
    initial_y = max_y

    X_vertices = [x_vertice for x_vertice in range(min_x, max_x, x_offset)]
    Y_vertices = [y_vertice for y_vertice in range(min_y, max_y, y_offset)]

    ######################################################

    new_dataframe = pd.DataFrame()

    serious_and_fatal_dataframe = data_frame[data_frame[casualty_name].isin(casualty_target_names)]

    for y_vertice in tqdm(Y_vertices):
        box_y_min = y_vertice
        box_y_max = y_vertice + y_offset

        serious_and_fatal_on_this_height = serious_and_fatal_dataframe[serious_and_fatal_dataframe[y_name].between(box_y_min, box_y_max)]

        if serious_and_fatal_on_this_height.empty: continue

        all_entries = data_frame[data_frame[y_name].between(box_y_min, box_y_max)]

        for x_vertice in X_vertices:
            box_x_min = x_vertice
            box_x_max = x_vertice + x_offset

            serious_and_fatal_on_this_box = serious_and_fatal_on_this_height[serious_and_fatal_on_this_height[x_name].between(box_x_min, box_x_max)]

            if serious_and_fatal_on_this_box.empty: continue

            all_entries = all_entries[all_entries[x_name].between(box_x_min, box_x_max)]

            new_dataframe = pd.concat([new_dataframe, all_entries])
            # print(len(new_dataframe))

    return new_dataframe

days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

## accepts an integer, no strings, if weekday is more than 6 - prints error and returns something else than a week day
def dayNameFromWeekday(weekday):
    if weekday>7:

        print('error');
        return 'an unknown day'
    return days[weekday-1]
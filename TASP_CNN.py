#!/usr/bin/env python
# coding: utf-8

# # Diagrama de flujo

# <center><img src="Data/Data_flow.png"/></center>

# # Métodos

# ## Carga Google Drive

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# ## Versión y especificación de directorios

# In[2]:


from datetime import datetime

MODEL_TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

GA_SCORES_PATH = 'GA_Scores/'
HYPERPARAMS_PATH = './hyperparams/'
WEIGHTS_PATH = './feature_weights/'
REPORTS_PATH = 'Reports/'
MODELS_PATH  = 'Models/'

HYPERPARAMS_EVOLUTON_PATH = './hyperparams_evolution'


# ## Importar Tensorflow

# In[3]:


# !pip install tensorflow-addons


# In[4]:


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import model_to_dot, plot_model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, BatchNormalization, Add, concatenate, Conv2DTranspose, Flatten


# In[5]:


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
get_ipython().system('nvidia-smi')


# ## Importador/Exportador JSON

# In[6]:


import json

def write_json(feature_vector, root_path, file_name):
    with open(root_path + file_name, 'w') as outfile:
        json.dump(feature_vector, outfile)

def load_json(root_path, file_name):
    with open(root_path + file_name) as json_file:
        data = json.load(json_file)

    return data


# ## Construcción de imágenes

# In[7]:


import numpy as np
import math
from matplotlib import pyplot as plt

def get_feature_matrix_indexes(sorted_feature_vector,matrix):  

    half_row = round((matrix.shape[0] - 1) / 2)
    half_column = round((matrix.shape[1] - 1) / 2)

    matrix_indexes = {}
    
    index = 0

    for parent_key in sorted_feature_vector:
        normalized_index = math.ceil(index/2)

        if (index % 2 != 0): # Impar
            current_row = half_row - normalized_index
        else: # Par
            current_row = half_row + normalized_index

        sorted_child_indexes = np.argsort(feature_vector[parent_key]['feature_weights'])[::-1]

        child_names   = np.array(feature_vector[parent_key]['feature_childs'])
        child_weights = np.array(feature_vector[parent_key]['feature_weights'])

        sorted_child_names   = child_names[sorted_child_indexes]
        sorted_child_weights = child_weights[sorted_child_indexes]

        position = 0
        for sorted_child_index in sorted_child_indexes:
            normalized_position = math.ceil(position/2)

            if (position % 2 != 0): # Impar
                current_column = half_column - normalized_position
            else: # Par
                current_column = half_column + normalized_position

            matrix_indexes[child_names[sorted_child_index]] = [current_row, current_column]
            position = position + 1 

        index = index + 1

    return matrix_indexes
    
def fv2gi(feature_vector):

    max_dimension = 0
    for key in feature_vector:
        childs_number = len(feature_vector[key]['feature_childs'])
        max_dimension = max(childs_number, max_dimension)
                
    matrix = np.zeros((max_dimension, max_dimension))

    weights_vector = []
    for parent_key in feature_vector:
        wpi = sum([float(child_weight) for child_weight in feature_vector[parent_key]['feature_weights']])
        feature_vector[parent_key]['wpi'] = wpi
        weights_vector.append(wpi)

   
    sorted_feature_vector = sorted(feature_vector.items(),
                                   key = lambda item: item[1]['wpi'],
                                   reverse = True)
     
    sorted_feature_vector = dict(sorted_feature_vector)

    
    matrix_indexes = get_feature_matrix_indexes(sorted_feature_vector, matrix)

    return matrix_indexes

# matrix_indexes = fv2gi(feature_vector)


# ## Construcción Feature Vector

# In[8]:


def fill_feature_vector(X_dataset,child_weights):
  # Obtenemos el set de columnas del dataset
  train_columns_set  = set(X_dataset.columns)

  for parent_feature in feature_vector.keys():
    # Obtiene el set de características hijas del padre actual
    # dict.fromleys para mantener el orden, un set desordena los valores
    feature_childs_set = dict.fromkeys(feature_vector[parent_feature]['feature_childs'])

    # Obtener el índice de las columnas del actual padre para acceder a los pesos del XGBoost
    index_feature_childs = X_dataset.columns.get_indexer(feature_childs_set)

    feature_vector[parent_feature]['feature_weights'] = list([str(child_weight) for child_weight in child_weights[index_feature_childs]])

  return feature_vector


# ## Normalización de datos

# In[9]:


from scipy.stats import zscore
import pandas as pd

def normalize_data(X_data):

    # Create a sample df
    normalized_df = X_data

    # Calculate the zscores and drop zscores into new column
    for column in normalized_df.columns:
        normalized_df[column] = zscore(normalized_df[column])
    
    return normalized_df


# ## Oversampling de datos

# In[10]:


from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import KMeansSMOTE

def oversample_data(X_data, Y_labels):

    oversampler = BorderlineSMOTE(kind='borderline-2')
    # oversampler = RandomOverSampler()
    # oversampler = KMeansSMOTE()
    X_oversampled, Y_oversampled = oversampler.fit_resample(X_data, Y_labels)

    print('********** After OverSampling **********')
    print('Slight: ', (Y_oversampled == 'Slight').sum())
    print('Serious:', (Y_oversampled == 'Serious').sum())
    print('Fatal:  ', (Y_oversampled == 'Fatal').sum())
    print('\n Total X: ', len(X_oversampled), ' Total Y: ', len(Y_oversampled))

    return X_oversampled, Y_oversampled


# ## Construcción de imágenes

# In[11]:


def build_gray_images(dataset, max_dimension, matrix_indexes):

    matrix_3d = np.zeros((max_dimension, max_dimension, len(dataset.index)))
    print(len(dataset.index))
    for feature, value in matrix_indexes.items():
        matrix_3d[value[0], value[1],] = dataset[feature]
        
    return matrix_3d


# ## Algoritmo genético

# ### Inicializar población

# In[12]:


def generate_individual():

    learningRate = round(random.uniform(0.01, 1), 2)
    maxDepth = int(random.randrange(1, 20, step= 1))
    minChildWeight = round(random.uniform(0.01, 15.0), 1)

    # learningRate = round(random.uniform(0.01, 1), 2)
    # nEstimators = random.randrange(100, 2000, step = 150)
    # maxDepth = int(random.randrange(1, 20, step= 1))
    # minChildWeight = round(random.uniform(0.01, 15.0), 1)
    # gammaValue = round(random.uniform(0.01, 10.0), 2)
    # subSample = round(random.uniform(0.01, 1.0), 2)
    # colSampleByTree = round(random.uniform(0.01, 1.0), 2)
    # regAlpha  = round(random.uniform(40,180), 1)
    # regLambda = round(random.uniform(0,1), 3)
    
    individual = [learningRate, maxDepth, minChildWeight]

    return individual

def initilialize_population(numberOfParents):
    population = []
    # space={'max_depth': hp.quniform("max_depth", 3, 25, 1),
#         'gamma': hp.uniform ('gamma', 1,9),
#         'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
#         'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1),
#         'n_estimators': hp.quniform('n_estimators', 100, 4000, 100),
#         'tree_method': 'gpu_hist'
#     }
    for i in range(numberOfParents):
        # learningRate[i] = round(random.uniform(0.01, 1), 2)
        # nEstimators[i] = random.randrange(100, 2000, step = 150)
        # maxDepth[i] = int(random.randrange(1, 20, step= 1))
        # minChildWeight[i] = round(random.uniform(0.01, 15.0), 1)
        # gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
        # subSample[i] = round(random.uniform(0.01, 1.0), 2)
        # colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)
        # regAlpha[i]  = round(random.uniform(40,180), 1)
        # regLambda[i] = round(random.uniform(0,1), 3)

        population.append(generate_individual())
    
    # population = np.concatenate((learningRate, nEstimators, maxDepth, minChildWeight, gammaValue, subSample, colSampleByTree, regAlpha, regLambda), axis= 1)
  
    return np.array(population)


# ### Fitness function

# In[13]:


from sklearn.metrics import f1_score

def fitness_f1score(y_true, y_pred):

    fitness = round((f1_score(y_true, y_pred, average='micro')), 4)

    return fitness # Train the data annd find fitness score


# ### Evaluación de población

# In[14]:


from xgboost import XGBClassifier

def train_population(population, dMatrixTrain, dMatrixTest, y_test):
    fScore = []
    for i in range(population.shape[0]):
        # param = { 'objective':'multi:softprob',
        #           'tree_method': 'gpu_hist',
        #           'num_class': 3,
        #           'learning_rate': population[i][0],
        #           'n_estimators': population[i][1], 
        #           'max_depth': int(population[i][2]), 
        #           'min_child_weight': population[i][3],
        #           'gamma': population[i][4], 
        #           'subsample': population[i][5],
        #           'colsample_bytree': population[i][6],
        #           'reg_alpha': population[i][7],
        #           'reg_lambda': population[i][8]
        #         }

        param = { 'objective':'multi:softprob',
                  'tree_method': 'gpu_hist',
                  'num_class': 3,
                  'learning_rate': population[i][0],
                  'max_depth': int(population[i][1]), 
                  'min_child_weight': population[i][2]
                }

        num_round = 100
        xgb.set_config(verbosity=0)
        xgbT = xgb.train(param,
                         dMatrixTrain,
                         num_round)

        preds = xgbT.predict(dMatrixTest)
        
        single_predictions = [np.argmax(pred) for pred in preds]
        # preds = preds > 0.5

        fScore.append(fitness_f1score(y_test, single_predictions))

    return fScore


# ### Selección de padres

# In[15]:


# Select parents for mating
def new_parents_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1])) # Create an array to store fittest parents.

    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId  = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = -1 # Set this value to negative, in case of F1-score, so this parent is not selected again


    return selectedParents


# ### Cruzamiento de población

# In[16]:


'''
Mate these parents to create children having parameters from these parents (we are using uniform crossover method)
'''
def crossover_uniform(parents, childrenSize):
    
    crossoverPointIndex  = np.arange(0, np.uint8(childrenSize[1]), 1, dtype= np.uint8) # get all the index
    crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]), np.uint8(childrenSize[1]/2)) # select half  of the indexes randomly
    crossoverPointIndex2 = np.array(list(set(crossoverPointIndex) - set(crossoverPointIndex1))) #select leftover indexes
    
    children = np.empty(childrenSize)
    
    '''
    Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values
    will be picked from the indexes, which were randomly selected above. 
    '''
    for i in range(childrenSize[0]):
        
        #find parent 1 index 
        parent1_index = i%parents.shape[0]
        #find parent 2 index
        parent2_index = (i+1)%parents.shape[0]
        #insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
        #insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]

    return children


# ### Mutación

# In[17]:


# def mutation(crossover, numberOfParameters):
#     # Define minimum and maximum values allowed for each parameterminMaxValue = np.zeros((numberOfParameters, 2))
#     minMaxValue = np.zeros((numberOfParameters, 2))

#     minMaxValue[0:]  = [0.01, 1.0]  # min/max learning rate
#     minMaxValue[1,:] = [10, 2000]   # min/max n_estimator
#     minMaxValue[2,:] = [1, 20]      # min/max depth
#     minMaxValue[3,:] = [0, 10.0]    # min/max child_weight
#     minMaxValue[4,:] = [0.01, 10.0] # min/max gamma
#     minMaxValue[5,:] = [0.01, 1.0]  # min/max subsample
#     minMaxValue[6,:] = [0.01, 1.0]  # min/max colsample_bytree
#     minMaxValue[7,:] = [40.0,180.0] # min/max reg_alpha
#     minMaxValue[8,:] = [0.0, 1.0]   # min/max reg_lambda
 
#     # Mutation changes a single gene in each offspring randomly.
#     mutationValue = 0
#     parameterSelect = np.random.randint(0, numberOfParameters, 1)

#     print(parameterSelect)
# #             if parameterSelect == 0: # learning_rate
# #                 # mutationValue = round(np.random.uniform(-0.2, 0.2), 2)
# #                 mutationValue = round(random.uniform(-0.3, 0.3), 2)
# #             if parameterSelect == 1: # max_depth
# #                 # mutationValue = np.random.randint(-3, 3, 1)
# #                 mutationValue = int(random.randrange(-4, 4, step= 1))
# #             if parameterSelect == 2: # min_child_weight
# #                 # mutationValue = round(np.random.uniform(5, 5), 2)
# #                 mutationValue = round(random.uniform(-5, 5), 1)
#     if parameterSelect == 0: # learning_rate
#         # mutationValue = round(np.random.uniform(-0.2, 0.2), 2)
#         mutationValue = round(random.uniform(-0.2, 0.2), 2)
#     if parameterSelect == 1: # n_estimators
#         mutationValue = np.random.randint(-150, 150, 1)
#         # mutationValue = random.randrange(100, 2000, step = 150)
#     if parameterSelect == 2: # max_depth
#         mutationValue = np.random.randint(-4, 4, 1)
#         # mutationValue = int(random.randrange(1, 20, step= 1))
#     if parameterSelect == 3: # min_child_weight
#         # mutationValue = round(np.random.uniform(5, 5), 2)
#         mutationValue = round(random.uniform(-5, 5), 1)
#     if parameterSelect == 4: #gamma
#         # mutationValue = round(np.random.uniform(-2, 2), 2)
#         mutationValue = round(random.uniform(0.01, 10.0), 2)
#     if parameterSelect == 5: # subsample
#         # mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
#         mutationValue = round(random.uniform(0.01, 1.0), 2)
#     if parameterSelect == 6: # colsample
#         # mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
#         mutationValue = round(random.uniform(0.5, 1.0), 2)
#     if parameterSelect == 7: # reg_alpha
#         # mutationValue = round(np.random.uniform(-20,20), 1)
#         mutationValue = round(random.uniform(40,180), 1)
#     if parameterSelect == 8: # reg_lambda
#         # mutationValue = round(np.random.uniform(-0.2,0.2), 3)
#         mutationValue = round(random.uniform(0,1), 3)
  
#     # Introduce mutation by changing one parameter, and set to max or min if it goes out of range
#     for idx in range(crossover.shape[0]):
#         crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue

#         if(crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
#             crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]

#         if(crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
#             crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]

#     return crossover


# In[18]:


def mutation(crossover, numberOfParameters):
    
    # MUTATION_PROBABILITY = 1/numberOfParameters
    
    MUTATION_PROBABILITY = 0.3

    # Define minimum and maximum values allowed for each parameterminMaxValue = np.zeros((numberOfParameters, 2))
    minMaxValue = np.zeros((numberOfParameters, 2))

    minMaxValue[0:]  = [0.01, 1.0]  # min/max learning rate
    minMaxValue[1,:] = [1, 15]      # min/max depth
    minMaxValue[2,:] = [0, 10.0]    # min/max child_weight
#     minMaxValue[0:]  = [0.01, 1.0]  # min/max learning rate
#     minMaxValue[1,:] = [100, 2000]  # min/max n_estimator
#     minMaxValue[2,:] = [1, 15]      # min/max depth
#     minMaxValue[3,:] = [0, 10.0]    # min/max child_weight
#     minMaxValue[4,:] = [0.01, 10.0] # min/max gamma
#     minMaxValue[5,:] = [0.01, 1.0]  # min/max subsample
#     minMaxValue[6,:] = [0.01, 1.0]  # min/max colsample_bytree
#     minMaxValue[7,:] = [40.0,180.0] # min/max reg_alpha
#     minMaxValue[8,:] = [0.0, 1.0]   # min/max reg_lambda
 
    for idx in range(crossover.shape[0]):
        # Mutation changes a single gene in each offspring randomly.

        
        mutation_probability = np.random.rand(1)

        while MUTATION_PROBABILITY > mutation_probability:
            
            mutationValue = 0
            parameterSelect = np.random.randint(0, numberOfParameters, 1)

            print(idx, parameterSelect)
            
            # if parameterSelect == 0: # learning_rate
            #     # mutationValue = round(np.random.uniform(-0.2, 0.2), 2)
            #     mutationValue = round(random.uniform(0.01, 1), 2)
            # if parameterSelect == 1: # n_estimators
            #     # mutationValue = np.random.randint(-150, 150, 1)
            #     mutationValue = random.randrange(100, 2000, step = 150)
            # if parameterSelect == 2: # max_depth
            #     # mutationValue = np.random.randint(-3, 3, 1)
            #     mutationValue = int(random.randrange(1, 20, step= 1))
            # if parameterSelect == 3: # min_child_weight
            #     # mutationValue = round(np.random.uniform(5, 5), 2)
            #     mutationValue = round(random.uniform(0.01, 15.0), 1)
            # if parameterSelect == 4: #gamma
            #     # mutationValue = round(np.random.uniform(-2, 2), 2)
            #     mutationValue = round(random.uniform(0.01, 10.0), 2)
            # if parameterSelect == 5: # subsample
            #     # mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
            #     mutationValue = round(random.uniform(0.01, 1.0), 2)
            # if parameterSelect == 6: # colsample
            #     # mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
            #     mutationValue = round(random.uniform(0.5, 1.0), 2)
            # if parameterSelect == 7: # reg_alpha
            #     # mutationValue = round(np.random.uniform(-20,20), 1)
            #     mutationValue = round(random.uniform(40,180), 1)
            # if parameterSelect == 8: # reg_lambda
            #     # mutationValue = round(np.random.uniform(-0.2,0.2), 3)
            #     mutationValue = round(random.uniform(0,1), 3)

            if parameterSelect == 0: # learning_rate
                # mutationValue = round(np.random.uniform(-0.2, 0.2), 2)
                mutationValue = round(random.uniform(-0.3, 0.3), 2)
            if parameterSelect == 1: # max_depth
                # mutationValue = np.random.randint(-3, 3, 1)
                mutationValue = int(random.randrange(-4, 4, step= 1))
            if parameterSelect == 2: # min_child_weight
                # mutationValue = round(np.random.uniform(5, 5), 2)
                mutationValue = round(random.uniform(-5, 5), 1)


            mutation_probability = np.random.rand(1)

            crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue

            if(crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
                crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]

            if(crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
                crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]

    return crossover


# ## Reshape de imágenes

# In[19]:


# Add one channel
# https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/

# Add one channel to gray images depending of the number of the data
def shape_images(X_data, gray_images):
  images = []

  for i in range(0,len(X_data)):
      original_matrix = gray_images[:,:,i]
      # print(original_matrix.shape)
      shaped_image = np.expand_dims(original_matrix, axis=2)
      # print(shaped_image.shape)
      images.append(shaped_image)
      # plt.matshow(shaped_image)

  return images


# ## One-Hot Encoder/Decoder

# In[20]:


def casualty_to_one_hot(Y_labels):

    transf = {
        'Slight': 0,
        'Serious': 1,
        'Fatal': 2
    }

    Y_labels.replace(transf, inplace = True)

    return tf.one_hot(Y_labels, 3)

def one_hot_to_casualty(Y_labels):

    transf = {
        0: 'Slight',
        1: 'Serious',
        2: 'Fatal'
    }   

    return Y_labels.replace(transf)


# ## Visualización de datos

# ### Matriz de correlación

# In[21]:


import seaborn as sns

def correlation_matrix(X_data):
    corrMatrix = X_data.corr()
    fig,ax  = plt.subplots(1,1,figsize=(15,10))
    sns.heatmap(corrMatrix, annot=True)
    plt.show()


# ### PCA

# In[22]:


from sklearn.decomposition import PCA

def pca(X_train_data, X_test_data):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_data)
    X_test_pca  = pca.transform(X_test_data)
    explained_variance = pca.explained_variance_ratio_

    figure_name = plt.figure(figsize=(15, 10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


# ### TSNE

# In[23]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def plot_TSNE(X_data, Y_data, n_components, output_file_name = None):

    X_data_scaled = StandardScaler().fit_transform(X_data)
    z_data = TSNE(n_components = n_components).fit_transform(X_data_scaled)

    # X_test_scaled = StandardScaler().fit_transform(X_test),
    # z_test = TSNE(n_components=2).fit_transform(X_test_scaled),

    palette = sns.color_palette('husl', 3)
    fig,ax  = plt.subplots(1,1,figsize=(15,10))
    sns.scatterplot(x = z_data[:,0],
                    y = z_data[:,1],
                    hue = Y_data,
                    palette = palette,
                    legend = 'full')

    if (output_file_name): plt.savefig('./Out/' + output_file_name)


# ### Autoencoder

# In[24]:


def autoencoder ():
    input_img = Input(shape=(25,))

    # definimos el encoder, que tendra una entrada de Input_img y una segunda capa con entrada de encoder1 y salida 3
    encoder1 = layers.Dense(15, activation='sigmoid')(input_img)
    encoder2 = layers.Dense(3, activation='sigmoid')(encoder1)

    # definimos el  decoder que tendra una entrada inicial de encoder3 y una salida de 128 y finalmete una capa de salida con los mismos que Input_img
    decoder1 = layers.Dense(15, activation='sigmoid')(encoder2)
    decoder2 = layers.Dense(25, activation='sigmoid')(decoder1)

    # this model maps an input to its reconstruction
    autoencoder = tf.keras.Model(inputs=input_img, outputs=decoder2)
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy') #se usan estos dos en estas arquitecturas
    
    return autoencoder


# ## TASP-CNN

# In[25]:


import tensorflow_addons as tfa

lr_init = 0.1
num_classes = 3

tasp_cnn = models.Sequential()
tasp_cnn.add(layers.Conv2D(256, (3, 3),strides=(1, 1), activation='relu', padding='same', input_shape=(5, 5, 1)))
tasp_cnn.add(layers.BatchNormalization())
tasp_cnn.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(3, 3, 256)))
tasp_cnn.add(layers.BatchNormalization())
tasp_cnn.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(3, 3, 256)))
tasp_cnn.add(layers.BatchNormalization())
tasp_cnn.add(layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=(3, 3, 256)))
tasp_cnn.add(layers.BatchNormalization())  
tasp_cnn.add(layers.Flatten())
tasp_cnn.add(layers.Dense(units=128))
tasp_cnn.add(layers.Dense(3, activation='softmax'))

tasp_cnn.compile(
    optimizer=Adam(learning_rate = lr_init, epsilon=1e-06),
    loss='categorical_crossentropy',
    metrics=[tfa.metrics.F1Score(num_classes = num_classes, average='micro', threshold=0.1)]
  )


# In[26]:


print('Done!')


# # Leeds Data

# ## Importación de datos

# In[27]:


# !conda install pandas --y


# In[28]:


import pandas as pd
# https://datamillnorth.org/dataset/road-traffic-accidents
# root_path = '/content/drive/Othercomputers/Mi portátil/Drive/Master UA/TFM/TFM_DATA/'

root_path = './Data/Leeds/'

file_path_2009 = './2009.csv'
file_path_2010 = './2010.csv'
file_path_2011 = './2011.csv'
file_path_2012 = './2012.csv'
file_path_2013 = './2013.csv'
file_path_2014 = './2014.csv'
file_path_2015 = './2015.csv'
file_path_2016 = './2016.csv'

file_2009 = pd.read_csv(root_path + file_path_2009, encoding = 'cp1252')
file_2010 = pd.read_csv(root_path + file_path_2010, encoding = 'cp1252')
file_2011 = pd.read_csv(root_path + file_path_2011, encoding = 'cp1252')
file_2012 = pd.read_csv(root_path + file_path_2012, encoding = 'cp1252')
file_2013 = pd.read_csv(root_path + file_path_2013, encoding = 'cp1252')
file_2014 = pd.read_csv(root_path + file_path_2014, encoding = 'cp1252')
file_2015 = pd.read_csv(root_path + file_path_2015, encoding = 'cp1252')
file_2016 = pd.read_csv(root_path + file_path_2016, encoding = 'cp1252')

###################### UNIÓN DE ARCHIVOS ######################

a = pd.concat([file_2009,file_2010])

file_2013 = clean_df = file_2013.loc[:, ~file_2013.columns.isin(['Casualty Class'])]
file_2013.set_axis(a.columns, axis=1, inplace=True)
                                             
file_2014 = clean_df = file_2014.loc[:, ~file_2014.columns.isin(['Casualty Class'])]
file_2014.set_axis(a.columns, axis=1, inplace=True)

# file_2015 = clean_df = file_2015.loc[:, ~file_2015.columns.isin(['Casualty Class'])]
file_2015.set_axis(a.columns, axis=1, inplace=True)
file_2016 = clean_df = file_2016.loc[:, ~file_2016.columns.isin(['Expr1'])]
file_2016.set_axis(a.columns, axis=1, inplace=True)

a = pd.concat([a, file_2011])
a = pd.concat([a, file_2012])
a = pd.concat([a, file_2013])
a = pd.concat([a, file_2014])
a = pd.concat([a, file_2015])
a = pd.concat([a, file_2016])

# a['1st Road Class'].value_counts()


# ## Limpieza de datos

# In[29]:


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

##################################
accident_date_replace = {
    'Dry': 1,
    'Wet / Damp': 2,
    'Snow': 3,
    'Frost / Ice': 4,
    'Flood': 5,
}
##################################

road_surface_replace = {
    'Dry': 1,
    'Wet / Damp': 2,
    'Snow': 3,
    'Frost/ Ice': 4,
    'Frost / Ice': 4,
    'Flood': 5,
    'Flood (surface water over 3cm deep)': 5,
    '5': 5
}

# La 5: "Darkness: street lighting unknown" no está presente en el paper, le hemos puesto un 5 porque sí #
lighting_conditions_replace = {
    'Daylight: street lights present': 1,
    'Darkness: no street lighting': 2,
    'Darkness: street lights present and lit': 3,
    'Darkness: street lights present but unlit': 4,
    'Darkness: street lighting unknown': 5,
    '5': 5
}

# La 8.2: "Unknown" no está presente en el paper, le hemos puesto un 8 porque sí (Other) #
weather_conditions_replace = {
    'Fine without high winds': 1,
    'Raining without high winds': 2,
    'Snowing without high winds': 3,
    'Fine with high winds': 4,
    'Raining with high winds': 5,
    'Snowing with high winds': 6,
    'Fog or mist – if hazard': 7,
    'Other': 8,
    'Unknown': 8
}

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

a['Time (24hr)'] = a['Time (24hr)'].mask(a['Time (24hr)'] < 600, 2)
a['Time (24hr)'] = a['Time (24hr)'].mask(a['Time (24hr)'] > 1800, 2)
a['Time (24hr)'] = a['Time (24hr)'].mask(a['Time (24hr)'].between(600, 1800), 1)
# print('Time (24hr):', a['Time (24hr)'].unique())
a.rename(columns={"Time (24hr)": "Accident Time"}, inplace = True)

###################### LIMPIEZA DE VALORES NULOS/DUPLICADOS ######################

clean_df = a.loc[:, ~a.columns.isin(['Accident Date', 'Reference Number'])]

clean_df['Weather Conditions'] = clean_df['Weather Conditions'].astype('int')
clean_df['Casualty Class']     = clean_df['Casualty Class'].astype('int')

clean_df = clean_df.drop_duplicates()
clean_df = clean_df.dropna()
clean_df = clean_df.reset_index(drop=True)

clean_df


# ## Split de datos

# In[30]:


# !conda install scikit-learn --y


# In[31]:


from sklearn.model_selection import train_test_split

Y = clean_df['Casualty Severity']

train, test = train_test_split(clean_df, test_size=0.2)
X_train = train.loc[:, ~train.columns.isin(['Casualty Severity'])]
Y_train = train['Casualty Severity']

X_test = test.loc[:, ~test.columns.isin(['Casualty Severity'])]
Y_test = test['Casualty Severity']


# ### Downsampling

# In[32]:


from sklearn.model_selection import train_test_split

Y = clean_df['Casualty Severity']

from sklearn.utils import resample

downsampled_train, downsampled_test = train_test_split(clean_df, test_size=0.2)

slight_data  = test[test['Casualty Severity'] == 'Slight']
serious_data = test[test['Casualty Severity'] == 'Serious']
fatal_data   = test[test['Casualty Severity'] == 'Fatal']

X_slight_downsampled = resample(slight_data,
                                replace = True,
                                n_samples = len(fatal_data))

X_serious_downsampled = resample(serious_data,
                                 replace = True,
                                 n_samples = len(fatal_data))

downsampled_dataset = pd.concat([X_slight_downsampled, X_serious_downsampled, fatal_data])

downsampled_train, downsampled_test = train_test_split(downsampled_dataset, test_size=0.2)


X_train_downsampled = downsampled_train.loc[:, ~downsampled_train.columns.isin(['Casualty Severity'])]
Y_train_downsampled = downsampled_train['Casualty Severity']

X_test_downsampled = downsampled_test.loc[:, ~downsampled_test.columns.isin(['Casualty Severity'])]
Y_test_downsampled = downsampled_test['Casualty Severity']


# In[33]:


# fv2gi(feature_vector)
# sorted(feature_vector.items(), key = lambda item: item[0][1])

# for item in feature_vector['Accident Features'].items():
#     print(item[1])

# feature_vector[parent_key].items()

# sorted(feature_vector['Accident Features'].items(), key = lambda item: item,
#                                reverse = True)

# print(feature_vector['Accident Features']['feature_weights'])

# fv = np.array(feature_vector['Accident Features']['feature_childs'])
# list(fv[indexes])


# ## Normalización de datos

# In[34]:


# !conda install -c conda-forge imbalanced-learn


# In[35]:


X_train = X_train.astype(int)
X_test  = X_test.astype(int)
X_train_downsampled = X_train_downsampled.astype(int)
X_test_downsampled  = X_test_downsampled.astype(int)

X_train = normalize_data(X_train)
X_test  = normalize_data(X_test)
X_train_downsampled = normalize_data(X_train_downsampled)
X_test_downsampled  = normalize_data(X_test_downsampled)


# ## Oversamplig de datos

# In[36]:


print('********** Before OverSampling **********')
print('Slight: ', (Y_train == 'Slight').sum())
print('Serious:', (Y_train == 'Serious').sum())
print('Fatal:  ', (Y_train == 'Fatal').sum())
print('\n Total X:', len(X_train), ' Total Y:', len(Y_train), '\n')

X_train, Y_train = oversample_data(X_train, Y_train)


# ## XGBoost

# In[37]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# ### Genético

# In[68]:


# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test  = sc.transform(X_test)

# XGboost Classifier
# model xgboost
# use xgboost API now

import xgboost as xgb
import random

Y_train_onehot = casualty_to_one_hot(Y_train)
Y_test_onehot  = casualty_to_one_hot(Y_test)

Y_train_downsampled_onehot = casualty_to_one_hot(Y_train_downsampled)
Y_test_downsampled_onehot  = casualty_to_one_hot(Y_test_downsampled)

numberOfParents = 50 # number of parents to start
numberOfParentsMating = 25 # Number of parents that will mate
numberOfParameters = 3  # Number of parameters that will be optimized
numberOfGenerations = 100 # Number of genration that will be created 

# Define the population size
populationSize = (numberOfParents, numberOfParameters) # initialize the population with randomly generated parameters

population = initilialize_population(numberOfParents) # Define an array to store the fitness  hitory
fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents]) # Define an array to store the value of each parameter for each parent and generation
populationHistory = np.empty([(numberOfGenerations+1)*numberOfParents, numberOfParameters]) # Insert the value of initial parameters in history

best_solution_history = np.empty([(numberOfGenerations), numberOfParameters+1])
populationHistory[0:numberOfParents, :] = population

xgbDMatrixTrain = xgb.DMatrix(data = X_train_downsampled, label = Y_train_downsampled)
xgbDMatrixTest  = xgb.DMatrix(data = X_test_downsampled,  label = Y_test_downsampled)
    
for generation in range(numberOfGenerations):
    print("This is number %s generation" % (generation))
    new_population = []
    # print(type(population[1]))
    # print(f'Population before resampling is  {population}')
    
    unique_individuals = np.unique(population, axis=0)
    
    new_individuals_to_create = numberOfParents - len(unique_individuals)
    
    for i in range(new_individuals_to_create):
        new_population.append(generate_individual())
    
    new_population = np.array(new_population)
    print(type(population[1]), population[1])
    # print(type(new_population[1]), new_population[1])
    if (new_individuals_to_create):
        population = np.concatenate((unique_individuals, new_population), axis=0)
    # print(type(population))
    print(f'Current population is {population}')
    print(f'Current population is {new_population}')

    
    # Train the dataset and obtain fitness
    fitnessValue = train_population(population = population,
                                    dMatrixTrain = xgbDMatrixTrain,
                                    dMatrixTest =  xgbDMatrixTest,
                                    y_test = Y_test_downsampled)

    fitnessHistory[generation,:] = fitnessValue
    # Best score in the current iteration
    max_score_index = np.argmax(fitnessHistory[generation,:])
    max_score_value = np.max(fitnessHistory[generation,:])
    max_score_solution = population[max_score_index]

    max_solution_with_score = []
    max_solution_with_score = np.append(max_score_solution, max_score_value)
    best_solution_history[generation] = max_solution_with_score

    print(f"Best F1 score in the this iteration = {max_score_value}, best solution {max_score_solution}") # Survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected
    
    parents = new_parents_selection(population = population,
                                    fitness = fitnessValue,
                                    numParents = numberOfParentsMating)
    
    # Mate these parents to create children having parameters from these parents (we are using uniform crossover)
    children = crossover_uniform(parents = parents,
                                 childrenSize = (populationSize[0] - parents.shape[0], numberOfParameters))
    
    # Add mutation to create genetic diversity
    children_mutated = mutation(children, numberOfParameters)
    
    '''
    We will create new population, which will contain parents that where selected previously based on the
    fitness score and rest of them  will be children
    '''
    population[0:parents.shape[0], :] = parents # Fittest parents
    population[parents.shape[0]:, :]  = children_mutated # Children
    
    populationHistory[(generation+1)*numberOfParents : (generation+1)*numberOfParents+ numberOfParents , :] = population # Store parent information
    
#Best solution from the final iteration

fitness = train_population(population = population,
                           dMatrixTrain = xgbDMatrixTrain,
                           dMatrixTest  = xgbDMatrixTest,
                           y_test = Y_test_downsampled)

fitnessHistory[generation+1, :] = fitness # index of the best solution
bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]

best_hyperparams = {}
best_hyperparams['eta'] = population[bestFitnessIndex][0]
best_hyperparams['max_depth'] = population[bestFitnessIndex][1]
best_hyperparams['min_child_weight'] = population[bestFitnessIndex][2]

# best_hyperparams['eta'] = population[bestFitnessIndex][0]
# best_hyperparams['n_estimators']  = population[bestFitnessIndex][1]
# best_hyperparams['max_depth'] = population[bestFitnessIndex][2]
# best_hyperparams['min_child_weight'] = population[bestFitnessIndex][3]
# best_hyperparams['gamma'] = population[bestFitnessIndex][4]
# best_hyperparams['subsample'] =  population[bestFitnessIndex][5]
# best_hyperparams['colsample_bytree'] =  population[bestFitnessIndex][6]
# best_hyperparams['reg_alpha'] =  population[bestFitnessIndex][7]
# best_hyperparams['reg_lambda'] =  population[bestFitnessIndex][8]


x_fitness = [np.max(fitnessHistory[i]) for i in range(0,fitnessHistory.shape[0])]

FILE_NAME = 'leeds_ga_' + MODEL_TIMESTAMP  + '.jpg'

plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(x_fitness)), x_fitness)
plt.savefig(GA_SCORES_PATH + FILE_NAME)


# In[80]:


FILE_NAME = 'leeds_ga_hyperparams_evolution_' + MODEL_TIMESTAMP  + '.jpg'

LEGEND_LABELS = ['Learning Rate', 'Max Depth', 'Min Child Weigth', 'Score']

plt.figure(figsize=(15, 8))
plt.plot(best_solution_history)
plt.legend(LEGEND_LABELS)
plt.savefig(HYPERPARAMS_EVOLUTON_PATH + FILE_NAME)


# ### Hiperparámetros

# #### Carga hiperparámetros

# In[ ]:


# FILE_NAME = 'madrid_hyperparams_v7.json'

# best_hyperparams = load_json(HYPERPARAMS_PATH, FILE_NAME)
# 0.04 2 6.5 --> 0.778


# #### Cálculo de Hiperparámetros

# In[ ]:


# Y_train_onehot = casualty_to_one_hot(Y_train)
# Y_test_onehot  = casualty_to_one_hot(Y_test)

# space={'max_depth': hp.quniform("max_depth", 3, 25, 1),
#         'gamma': hp.uniform ('gamma', 1,9),
#         'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
#         'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1),
#         'n_estimators': hp.quniform('n_estimators', 100, 4000, 100),
#         'tree_method': 'gpu_hist'
#     }

# def objective(space):
#     clf = XGBClassifier(n_estimators = int(space['n_estimators']),
#                         max_depth = int(space['max_depth']),
#                         gamma = space['gamma'],
#                         reg_alpha = int(space['reg_alpha']),
#                         min_child_weight = int(space['min_child_weight']),
#                         colsample_bytree = int(space['colsample_bytree']),
#                         tree_method = space['tree_method']
#                        )
    
#     evaluation = [(X_train, Y_train), (X_test, Y_test)]
    
#     clf.fit(X_train, Y_train,
#             eval_set = evaluation, eval_metric = "auc",
#             early_stopping_rounds = 10, verbose = False)
            
    
#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(Y_test, pred>0.5)
#     print ("SCORE:", accuracy)
#     return {'loss': -accuracy, 'status': STATUS_OK }


# trials = Trials()

# best_hyperparams = fmin(fn = objective,
#                         space = space,
#                         algo = tpe.suggest,
#                         max_evals = 100,
#                         trials = trials)


# #### Escritura hiperparámetros

# In[ ]:


FILE_NAME = 'leeds_hyperparams' + MODEL_TIMESTAMP + '.json'

write_json(best_hyperparams, HYPERPARAMS_PATH, FILE_NAME)
print(best_hyperparams)


# ### Pesos de características

# #### Carga definitiva/auxiliar de pesos

# In[ ]:


# FILE_NAME = 'leeds_calculated_weights.json'
FILE_NAME = 'leeds_weights2022-04-14-10:55:09.json'

feature_vector = load_json(WEIGHTS_PATH, FILE_NAME)


# #### Cálculo de pesos de caracetrísticas

# In[ ]:


from numpy import loadtxt
from xgboost import XGBClassifier,XGBRanker
from matplotlib import pyplot
from xgboost import plot_importance

xgboost = XGBClassifier(best_hyperparams, tree_method = 'gpu_hist')

xgboost.fit(X_train, Y_train)

child_weights  = np.array(xgboost.feature_importances_)
feature_vector = fill_feature_vector(X_train, child_weights)


# #### Visualización pesos calculados

# In[ ]:


FILE_NAME = 'leeds_figure_weights' + MODEL_TIMESTAMP + '.jpg'

print(xgboost.get_booster().get_score(importance_type= 'weight'))
plt.figure(figsize=(10, 5))
plt.barh(X_train.columns, xgboost.feature_importances_)
plt.savefig(WEIGHTS_PATH + FILE_NAME)

for column, weight in zip(X_train.columns,xgboost.feature_importances_):
  print(column, weight)


# #### Escritura de pesos de características

# In[ ]:


matrix_indexes = fv2gi(feature_vector)

FILE_NAME = 'leeds_weights' + MODEL_TIMESTAMP + '.json'
# FILE_NAME = 'leeds_default_weights.json'

write_json(feature_vector, WEIGHTS_PATH, FILE_NAME)


# ### Cálculo índices de matriz

# In[ ]:


matrix_indexes = fv2gi(feature_vector)


# ## Construcción de imágenes

# In[ ]:


train_bgi = build_gray_images(X_train, 5, matrix_indexes)
test_bgi  = build_gray_images(X_test, 5, matrix_indexes)

pd.DataFrame(train_bgi[:,:,1057])


# ## Reshape de imágenes

# In[ ]:


train_images = shape_images(X_data = X_train,
                            gray_images = train_bgi)
test_images  = shape_images(X_data = X_test,
                            gray_images = test_bgi)


# In[ ]:


plt.gray()
for i in range(0,3):
    plt.figure(figsize=(3, 3))
    plt.imshow(train_bgi[:,:,i])
    plt.show()


# In[ ]:


# !conda install scikit-image


# In[ ]:


# input_shape = (5, 5)

array_train_images = np.asarray(train_images)
array_test_images  = np.asarray(test_images)


# 

# In[ ]:


# !conda install -c conda-forge tensorflow 


# In[ ]:


######### EN TERMINAL #########
# !pip install tensorflow-addons


# ## One-Hot

# In[ ]:


Y_train_onehot = casualty_to_one_hot(Y_train)
Y_test_onehot  = casualty_to_one_hot(Y_test)


# ## Visualización de datos

# In[ ]:


# !conda install seaborn


# ### Matriz de correlación

# In[ ]:


# correlation_matrix(X_test)


# ### PCA

# In[ ]:


# pca(X_train, X_test)


# ### TSNE

# In[ ]:


# output_file_name = './2d_test_tsne.jpg'
# plot_TSNE(X_test, Y_test, n_components=2)

# output_file_name = './2d_train_tsne.jpg'
# plot_TSNE(X_train, Y_train, n_components=2)

# output_file_name = './3d_test_tsne.jpg'
# plot_TSNE(X_test, Y_test, n_components=3)

# output_file_name = './3d_train_tsne.jpg'
# plot_TSNE(X_train, Y_train, n_components=3)


# ### Autoencoder

# #### Entrenamiento

# In[ ]:


X_train = array_train_images
X_test = array_test_images

X_train = X_train.reshape(len(array_train_images), 25)
X_test  = X_test.reshape(len(X_test), 25)

# autoencoder().fit(X_train, X_train,
#                 epochs=15,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_data=(X_test, X_test))


# #### Visualización

# In[ ]:


# # create encoder model
# encoder = Model(inputs=input_img, outputs=encoder2)
# encoder.summary()
# # create decoder model
# encoded_input = Input(shape=(3,))
# #lo que hace aqui es quedarse con las capas que corresponden al decodificador
# decoder_layer1 = autoencoder.layers[-2]
# decoder_layer2 = autoencoder.layers[-1]
# decoder = Model(inputs=encoded_input, outputs=decoder_layer2(decoder_layer1(encoded_input)))
# decoder.summary()
# # si miramos la salida, son simetricos el uno respecto al otro
# # encoder va de input a 3 y decoder de 3 a input

# # get latent vector for visualization
# latent_vector = encoder.predict(X_test)
# # get decoder output to visualize reconstructed image
# reconstructed_imgs = decoder.predict(latent_vector)


# # visualize in 3D plot
# from pylab import rcParams
# from mpl_toolkits.mplot3d import Axes3D

# rcParams['figure.figsize'] = 10, 8

# fig = plt.figure(1)
# ax = Axes3D(fig)

# xs = latent_vector[:, 0]
# ys = latent_vector[:, 1]
# zs = latent_vector[:, 2]

# color=['red','green','blue']

# for x, y, z, label in zip(xs, ys, zs, Y_test):
#     c = color[int(label)]
#     ax.text(x, y, z, label, backgroundcolor=c)
    
# ax.set_xlim(xs.min(), xs.max())
# ax.set_ylim(ys.min(), ys.max())
# ax.set_zlim(zs.min(), zs.max())

# # plt.show()

# # X_test_encoded = encoder.predict(X_test, batch_size=32)
# # plt.figure(figsize=(6, 6))
# # plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], c=Y_test)
# # plt.colorbar()
# # plt.show()


# In[ ]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def plot_TSNE(X_data, Y_data, n_components, output_file_name=None):
    X_data_scaled = StandardScaler().fit_transform(X_data)
    z_data = TSNE(n_components=n_components).fit_transform(X_data_scaled)
    # X_test_scaled = StandardScaler().fit_transform(X_test)
    # z_test = TSNE(n_components=2).fit_transform(X_test_scaled)

    palette = sns.color_palette("hls", 3)
    fig,ax  = plt.subplots(1,1,figsize=(15,10))
    sns.scatterplot(x = z_data[:,0],
                    y = z_data[:,1],
                    hue = Y_data,
                    palette = palette,
                    legend = 'full')

    if (output_file_name): plt.savefig('./Out/' + output_file_name)

# output_file_name = './2d_test_tsne.jpg'
# plot_TSNE(X_test, Y_test, n_components=2)

# output_file_name = './2d_train_tsne.jpg'
# plot_TSNE(X_train, Y_train, n_components=2)

# output_file_name = './3d_test_tsne.jpg'
# plot_TSNE(X_test, Y_test, n_components=3)

# output_file_name = './3d_train_tsne.jpg'
# plot_TSNE(X_train, Y_train, n_components=3)


# ## TASP-CNN

# ### Entrenamiento

# In[ ]:


input_train_shape = (len(array_train_images), 5, 5, 1)
input_test_shape = (len(array_test_images), 5, 5, 1)

array_train_images.reshape(input_train_shape)
array_test_images.reshape(input_test_shape)

history = tasp_cnn.fit(array_train_images, Y_train_onehot,
                    batch_size = 128, epochs = 100, shuffle = True,
                    validation_data = (array_test_images, Y_test_onehot))

# history


# ### Escritura del modelo

# In[ ]:


tasp_cnn.save(MODELS_PATH + 'leeds_' + MODEL_TIMESTAMP + '.h5')


# ### Carga de modelo pre-entrenado

# In[ ]:


# tasp_cnn = tf.keras.models.load_model(MODELS_PATH + 'leeds_model_bayesian' + MODEL_TIMESTAMP + '.h5')


# ### Resultados

# In[ ]:


from sklearn.metrics import classification_report

Y_test_labels = one_hot_to_casualty(Y_test)

########################################################################

F1_SCORE_PATH = 'F1scores/'
F1_SCORE_NAME = 'leeds_f1_score' + MODEL_TIMESTAMP

## Plot history: F1 SCORE
figure_name = plt.figure(figsize=(20, 10))
plt.plot(history.history['f1_score'], label='F1 score (training data)')
plt.plot(history.history['val_f1_score'], label='F1 score (validation data)')
plt.title('F1 score')
plt.ylabel('F1 score value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig(F1_SCORE_PATH + F1_SCORE_NAME + '.jpg')
plt.show()

print(history)

########################################################################

# evaluate the network
print("[INFO] evaluating network...")
predictions = tasp_cnn.predict(x=array_test_images, batch_size=128)

report = classification_report(tf.argmax(Y_test_onehot, axis=1),
                               predictions.argmax(axis=1),
                               target_names = Y_test_labels.unique(),
                               output_dict = True)

REPORT_NAME  = 'leeds_report' + MODEL_TIMESTAMP + '.csv'

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH + REPORT_NAME, index= True)

report_df


# In[ ]:


# import tensorflow as tf
# from sklearn.metrics import classification_report

# predictions = xgboost.predict(X_test)
# predictions


# In[ ]:


# prediction_string = [str(prediction) for prediction in predictions]
# pd.DataFrame(prediction_string).value_counts()


# # Madrid Data

# ## Importación de datos

# - [Web Dataset](https://datos.madrid.es/portal/site/egob/menuitem.c05c1f754a33a9fbe4b2e4b284f1a5a0/?vgnextoid=7c2843010d9c3610VgnVCM2000001f4a900aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD&vgnextfmt=default)
# 
# - [Web documentación](https://datos.madrid.es/FWProjects/egob/Catalogo/Seguridad/Ficheros/Estructura_DS_Accidentes_trafico_desde_2019.pdf)
# 
# 

# In[ ]:


import pandas as pd

# root_path = '/content/drive/Othercomputers/Mi portátil/Drive/Master UA/TFM/Incidentes de Trafico/Datasets/Madrid/'

root_path = './Data/Madrid/'

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


# ### Calcular Vehículos implicados

# A partir del número de expediente (un mismo expediente en varias filas quiere decir que se trata del mismo accidente) se hace un `groupby` a partir de él. Como el atributo `positiva_alcohol` no tiene valores nulos en ninguna de las filas, hacemos un conteo a partir de él y se asigna a una nueva columna `positiva_alcohol_rename` que posteriormente será renombrada como `vehiculos_implicados`

# In[ ]:


data_frame = data_frame.join(data_frame.groupby('num_expediente')['positiva_alcohol'].count(), on='num_expediente', rsuffix='_rename')
data_frame.rename(columns={"positiva_alcohol_rename": "vehiculos_implicados"}, errors="raise", inplace=True)
data_frame = data_frame.reset_index(drop=True)
# data_frame.localizacion.unique()[:1000]


# ## Limpieza de datos

# ### Clasificación de carreteras

# In[ ]:


######################### SIGUIENTE CELDA #########################

# Unclassified: Carreteras locales sin destino definido. Sin embargo, los destinos locales pueden estar señalizados a lo largo de ellos.
# A, A(M) y Motorway lo mismo?
# B:            De carácter regional y utilizado para conectar zonas de menor importancia.
#               Por lo general, se muestran de color marrón o amarillo en los mapas y tienen las mismas señales blancas que las rutas de clase A que no son primarias.
#               Si la ruta es primaria, como la B6261, se mostrará igual que una ruta Clase A primaria.
#               ¿Carretera como tal?

# C:            Designaciones de autoridades locales para rutas dentro de su área con fines administrativos.
#               Estas rutas no se muestran en mapas de carreteras a pequeña escala, pero se sabe que ocasionalmente aparecen en las señales de tráfico.

# Unclassified


regex = {}
regex['paseo_regex'] = 'PASEO|paseo'
regex['parque_regex'] = 'PARQUE|PQUE'
regex['ronda_regex'] = 'RONDA'
regex['puerta_regex'] = 'PUERTA|PTA|Puerta'
regex['puente_regex'] = 'PNTE|PUENTE'
regex['plaza_regex'] = 'PLAZA|PZA'
regex['camino_regex']= 'CMNO|CAMINO'
regex['bulevard_regex'] = 'BULE'
regex['travesia_regex'] = 'TRVA'
regex['cuesta_regex'] = 'CUSTA|CUESTA'
regex['rotonda_regex'] = 'GTA|gta|GLORIETA|glorieta|ROTONDA'
regex['aeropuerto_regex'] ='AEROPUERTO|AEROP'
regex['calzada_regex'] = 'CALZADA'
regex['poligono_regex'] ='POLIGONO'
regex['highway_regex'] = 'AUTOV.|autovia|A-|M-|M 30|m 30|A\\d|M 23|M23' # A,A(M),Motorway
regex['road_regex'] = 'CTRA.|CARRETERA|carretera|CRA.' # B
regex['avenida_regex'] = 'AVDA|AV|AVENIDA|AVDA|Avda.|avenida'
regex['calle_regex']  = 'CALL.|Calle|CALLE|c/|C/|C.|calle'

data_frame['tipo_via'] = 'N/A'

for index,regex_values in enumerate(regex.values()):
    print(regex_values)
    regex_indexes = data_frame[data_frame.localizacion.str.contains(regex_values,  case = True, regex=True)].index
    print(len(regex_indexes))
    data_frame.iloc[regex_indexes, data_frame.columns.get_loc('tipo_via')] = str(index)
    data_frame.iloc[regex_indexes, data_frame.columns.get_loc('localizacion')] = str(index)
    
    
    
# street_indexes  = data_frame[data_frame.localizacion.str.contains('CALL.|Calle|CALLE|c/|C/|C.|calle', case = True, regex=True)].index
# highway_indexes = data_frame[data_frame.localizacion.str.contains(highway_regex, case = True, regex=True)].index
# road_indexes    = data_frame[data_frame.localizacion.str.contains(road_regex, case = True, regex=True)].index
# # avenue_indexes  = data_frame[data_frame.localizacion.str.contains(avenue_regex,  case = True, regex=True)].index
# # ride_indexes    = data_frame[data_frame.localizacion.str.contains(ride_regex, case = True, regex=True)].index

# data_frame['tipo_via'] = 'N/A'

# data_frame.iloc[street_indexes,  data_frame.columns.get_loc('tipo_via')] = 'Unclassified'
# data_frame.iloc[highway_indexes, data_frame.columns.get_loc('tipo_via')] = 'A'
# data_frame.iloc[road_indexes, data_frame.columns.get_loc('tipo_via')] = 'B'
# # data_frame.iloc[ride_indexes, data_frame.columns.get_loc('tipo_via')] = 'AVENIDA'
# # data_frame.iloc[avenue_indexes,  data_frame.columns.get_loc('tipo_via')] = 'AVENIDA'


# data_frame.iloc[highway_indexes, data_frame.columns.get_loc('localizacion')] = 1
# data_frame.iloc[road_indexes, data_frame.columns.get_loc('localizacion')] = 2
# data_frame.iloc[street_indexes,  data_frame.columns.get_loc('localizacion')] = 3
# # data_frame.iloc[avenue_indexes,  data_frame.columns.get_loc('localizacion')] = '3'
# # data_frame.iloc[ride_indexes, data_frame.columns.get_loc('localizacion')] = '5'

positive_drug_indexes = data_frame[data_frame.positiva_droga == 1].index
data_frame.iloc[positive_drug_indexes, data_frame.columns.get_loc('positiva_alcohol')] = 'S'

data_frame = data_frame[~(data_frame.tipo_via == 'N/A')]
# # print(data_frame.localizacion.unique())


# In[ ]:


# ######################### SIGUIENTE CELDA #########################

# # Unclassified: Carreteras locales sin destino definido. Sin embargo, los destinos locales pueden estar señalizados a lo largo de ellos.
# # A, A(M) y Motorway lo mismo?
# # B:            De carácter regional y utilizado para conectar zonas de menor importancia.
# #               Por lo general, se muestran de color marrón o amarillo en los mapas y tienen las mismas señales blancas que las rutas de clase A que no son primarias.
# #               Si la ruta es primaria, como la B6261, se mostrará igual que una ruta Clase A primaria.
# #               ¿Carretera como tal?

# # C:            Designaciones de autoridades locales para rutas dentro de su área con fines administrativos.
# #               Estas rutas no se muestran en mapas de carreteras a pequeña escala, pero se sabe que ocasionalmente aparecen en las señales de tráfico.

# # Unclassified
# street_regex  = ('CALL.|Calle|CALLE|c/|C/|C.|calle|'
#                  'AVDA|AV|AVENIDA|AVDA|avenida|Avda.|'
#                  'PASEO|paseo|'
#                  'PARQUE|PQUE|'
#                  'RONDA|'
#                  'PUERTA|PTA|Puerta|'
#                  'PNTE|PUENTE|'
#                  'PLAZA|PZA|'
#                  'CMNO|CAMINO|'
#                  'BULE|'
#                  'TRVA|'
#                  'CUSTA|CUESTA|'
#                  'GTA|gta|GLORIETA|glorieta|ROTONDA|'
#                  'AEROPUERTO|AEROP'
# )

# highway_regex = 'AUTOV.|autovia|A-|M-|M 30|m 30|A\\d|M 23|M23' # A,A(M),Motorway
# road_regex = 'CTRA.|CARRETERA|carretera|CRA.|CALZADA|POLIGONO' # B

# street_indexes  = data_frame[data_frame.localizacion.str.contains(street_regex,  case = True, regex=True)].index
# highway_indexes = data_frame[data_frame.localizacion.str.contains(highway_regex, case = True, regex=True)].index
# road_indexes    = data_frame[data_frame.localizacion.str.contains(road_regex, case = True, regex=True)].index
# # avenue_indexes  = data_frame[data_frame.localizacion.str.contains(avenue_regex,  case = True, regex=True)].index
# # ride_indexes    = data_frame[data_frame.localizacion.str.contains(ride_regex, case = True, regex=True)].index

# data_frame['tipo_via'] = 'N/A'

# data_frame.iloc[street_indexes,  data_frame.columns.get_loc('tipo_via')] = 'Unclassified'
# data_frame.iloc[highway_indexes, data_frame.columns.get_loc('tipo_via')] = 'A'
# data_frame.iloc[road_indexes, data_frame.columns.get_loc('tipo_via')] = 'B'
# # data_frame.iloc[ride_indexes, data_frame.columns.get_loc('tipo_via')] = 'AVENIDA'
# # data_frame.iloc[avenue_indexes,  data_frame.columns.get_loc('tipo_via')] = 'AVENIDA'


# data_frame.iloc[highway_indexes, data_frame.columns.get_loc('localizacion')] = 1
# data_frame.iloc[road_indexes, data_frame.columns.get_loc('localizacion')] = 2
# data_frame.iloc[street_indexes,  data_frame.columns.get_loc('localizacion')] = 3
# # data_frame.iloc[avenue_indexes,  data_frame.columns.get_loc('localizacion')] = '3'
# # data_frame.iloc[ride_indexes, data_frame.columns.get_loc('localizacion')] = '5'

# # positive_drug_indexes = data_frame[data_frame.positiva_droga == 1].index
# # data_frame.iloc[positive_drug_indexes, data_frame.columns.get_loc('positiva_alcohol')] = 'S'

# data_frame = data_frame[~(data_frame.tipo_via == 'N/A')]
# # print(data_frame.localizacion.unique())


# Consideraciones:
# 
# - Los patinetes se han considerado como ciclomotres de menos de 50cc.
# - Las furgonetas se consideran como vehículos de menos de 3.5 toneladas.
# - Maquinaria de obras se considera la misma tipología que maquinaria agrícola.
# - Cuadriciclos ligeros y no ligeros se consideran como `Motorcycle-Unknown CC`.
# - Patinetes y Vehículos de Mobilidad Urbana se consideran como `Mobility Scooters`.
# - `Vehículo articulado` se considera como un vehículo de más de 7.5 toneladas.

# In[ ]:


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
    'Atropello a persona': 6,
    'Colisión frontal': 7,
    'Otro': 8,
    'Solo salida de la vía': 9,
    'Vuelco': 10,
    'Atropello a animal': 11,
    'Despeñamiento': 12
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

### OJO QUE ESTAMOS REPLICANDO LA ESTRUCTURA DEL DATASET DE LEEDS
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
print('Estado meteorológico: \n', data_frame['estado_meteorológico'].value_counts())

data_frame['tipo_vehiculo'].replace(type_of_vehicle_replace, inplace = True)
print('Tipo vehículo: \n', data_frame['tipo_vehiculo'].value_counts())

data_frame['tipo_persona'].replace(casualty_class_replace, inplace = True)
print('Tipo de persona: \n', data_frame['tipo_persona'].value_counts())

data_frame['sexo'].replace(sex_of_casualty_replace, inplace = True)
print('Sexo: \n', data_frame['sexo'].value_counts())

data_frame['positiva_alcohol'].replace(alcohol_replace, inplace = True)
print('Positivo Alcohol: \n', data_frame['positiva_alcohol'].value_counts())

data_frame['lesividad'].replace(accident_class_replace, inplace = True)
print('Gravedad: \n', data_frame['lesividad'].value_counts())

data_frame['rango_edad'].replace(age_replace, inplace = True)
print('Edad: \n', data_frame['rango_edad'].value_counts())

data_frame.hora = data_frame.hora.mask(pd.to_datetime(data_frame.hora) < '06:00:00', 2)
data_frame.hora = data_frame.hora.mask(pd.to_datetime(data_frame.hora) > '18:00:00', 2)
data_frame.hora = data_frame.hora.mask(pd.to_datetime(data_frame.hora).between('06:00:00', '18:00:00'), 1)
print('hora:', data_frame['hora'].value_counts())

district_replace = {}
for index,distrito in enumerate(data_frame.distrito.unique()):
  if not pd.isna(distrito): district_replace[distrito] = int(index)

accident_type_replace = {}
for index,accident_type in enumerate(data_frame.tipo_accidente.unique()):
    if not pd.isna(accident_type): accident_type_replace[accident_type] = int(index)

data_frame['distrito'].replace(district_replace, inplace = True)
print('Distrito: \n', data_frame['distrito'].value_counts())

data_frame['tipo_accidente'].replace(accident_type_replace, inplace = True)
print('Tipo Accidente: \n', data_frame['tipo_accidente'].value_counts())

# Eliminamos aquellas lesividades desconocidas i.e. 77.
data_frame = data_frame[data_frame.lesividad != 77]


# ### Coordenadas UTM a números enteros

# Las coordenadas UTM son coordenads que están expresadas en término de X e Y partiendo de la base de que parten desde una determinada localización. Estas coordenadas constan de una parte entera y una decimal.
# 
# En este dataset el formato que presentan estas coordenadas pueden ser de tres tipos:
# 
# - **XXX.XXX.XXX**: en este caso los seis primeros dígitos forman la parte entera y los tres útlimos la parte decimal.
# - **XXXXXX,XX**: los seis primeros dígitos indican la parte entera, mientras que tras la coma aparecen dos dígitos de la parte decimal que habrá que completar añadiendo uno más.
# - **XXXXXX**: indican la parte entera, sin contar con la parte decimal.
# 
# Por lo que el objetivo es estandarizar todos los formatos convirtiendo cada una de las coordenadas a un número entero, siendo necesario tratar con cada una de las casuísticas para añadir ceros a la derecha en caso de que falten para que cada una de las coordenadas tenga la misma longitud.

# In[ ]:


# Todos las comas a puntos

import re

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


# ### Renombrado y eliminación de columnas

# In[ ]:


COLUMNS_TO_REMOVE = ['num_expediente', 'fecha', 'tipo_via', 'numero', 'positiva_droga', 'coordenada_x_utm', 'coordenada_y_utm', 'positiva_droga']
data_frame = data_frame.loc[:, ~data_frame.columns.isin(COLUMNS_TO_REMOVE)]

data_frame.rename(columns={"localizacion": "tipo_carretera"}, errors="raise", inplace=True)
data_frame.rename(columns={"processed_x_utm": "coordenada_x_utm"}, errors="raise", inplace=True)
data_frame.rename(columns={"processed_y_utm": "coordenada_y_utm"}, errors="raise", inplace=True)
data_frame.rename(columns={"positiva_alcohol": "drogas_alcohol_positivo"}, errors="raise", inplace=True)

data_frame = data_frame.drop_duplicates()
data_frame = data_frame.dropna()
data_frame = data_frame.reset_index(drop=True)


# In[ ]:


# X_data_frame = data_frame.loc[:, ~data_frame.columns.isin(['lesividad'])]
# Y_data_frame = data_frame['lesividad']

# X_data_frame  = X_data_frame.astype(int)


# ## Split de datos

# In[ ]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(data_frame, test_size=0.2)
X_train = train.loc[:, ~train.columns.isin(['lesividad'])]
Y_train = train['lesividad']

X_test = test.loc[:, ~test.columns.isin(['lesividad'])]
Y_test = test['lesividad']


# In[ ]:


# # FILE_NAME = 'madrid_calculated_weights.json'
# FILE_NAME = 'madrid_adapted_leeds_default_weights.json'

# feature_vector = load_json(WEIGHTS_PATH, FILE_NAME)
# display(feature_vector)


# In[ ]:



# feature_vector = {}

# feature_vector['Accident Features'] = {
#     'feature_childs': ['coordenada_x_utm', 'coordenada_y_utm', 'distrito', 'hora', 'vehiculos_implicados'],
#     'feature_weights': [0.165774538, 0.171530785, 0.082228259, 0.047771472, 0.060763375]
# } 

# feature_vector['Roadway Features'] = {
#     'feature_childs': ['tipo_accidente'], # Road Surface 
#     'feature_weights': [0.07036541]
# }

# feature_vector['Environmental Features'] = {
#     'feature_childs': ['estado_meteorológico'],
#     'feature_weights': [0.04354843]
# }

# feature_vector['Vehicle Features'] = {
#     'feature_childs': ['tipo_vehiculo'],
#     'feature_weights': [0.126314657]
# }

# feature_vector['Casualty Features'] = {
#     'feature_childs': ['tipo_persona', 'sexo', 'rango_edad', 'drogas_alcohol_positivo'],
#     'feature_weights': [0.067057589, 0.049116389, 0.095220163, 0.059951354]
# } 
# matrix_indexes = fv2gi(feature_vector)

# # {'Accident Features': {'feature_childs': ['Easting',
# #    'Northing',
# #    '1st Road Class',
# #    'Accident Time',
# #    'Number of Vehicles'],
# #   'feature_weights': [0.165774538,
# #    0.171530785,
# #    0.082228259,
# #    0.047771472,
# #    0.060763375],
# #   'wpi': 0.528068429},
# #  'Roadway Features': {'feature_childs': ['Road Surface'],
# #   'feature_weights': [0.048847406],
# #   'wpi': 0.048847406},
# #  'Environmental Features': {'feature_childs': ['Lighting Conditions',
# #    'Weather Conditions'],
# #   'feature_weights': [0.041826936, 0.04354843],
# #   'wpi': 0.08537536600000001},
# #  'Vehicle Features': {'feature_childs': ['Type of Vehicle'],
# #   'feature_weights': [0.126314657],
# #   'wpi': 0.126314657},
# #  'Casualty Features': {'feature_childs': ['Casualty Class',
# #    'Sex of Casualty',
# #    'Age of Casualty'],
# #   'feature_weights': [0.067057589, 0.049116389, 0.095220163],
# #   'wpi': 0.211394141}}
# feature_vector


# ## Normalización de datos

# In[ ]:


X_train = X_train.astype(int)
X_test  = X_test.astype(int)

X_train = normalize_data(X_train)
X_test  = normalize_data(X_test)


# ## Oversampling de datos

# In[ ]:


print('********** Before OverSampling **********')
print('Slight: ', (Y_train == 'Slight').sum())
print('Serious:', (Y_train == 'Serious').sum())
print('Fatal:  ', (Y_train == 'Fatal').sum())
print('\n Total X:', len(X_train), ' Total Y:', len(Y_train), '\n')

X_train, Y_train = oversample_data(X_train, Y_train)


# ## Downsampling de datos

# In[ ]:


from sklearn.model_selection import train_test_split

from sklearn.utils import resample

slight_data  = test[test['lesividad'] == 'Slight']
serious_data = test[test['lesividad'] == 'Serious']
fatal_data   = test[test['lesividad'] == 'Fatal']

X_slight_downsampled = resample(slight_data,
                                replace = True,
                                n_samples = len(fatal_data))

X_serious_downsampled = resample(serious_data,
                                 replace = True,
                                 n_samples = len(fatal_data))

downsampled_test = pd.concat([X_slight_downsampled, X_serious_downsampled, fatal_data])

# downsampled_train, downsampled_test = train_test_split(downsampled_dataset, test_size=0.2)


# X_train_downsampled = downsampled_train.loc[:, ~downsampled_train.columns.isin(['lesividad'])]
# Y_train_downsampled = downsampled_train['lesividad']

X_test_downsampled = downsampled_test.loc[:, ~downsampled_test.columns.isin(['lesividad'])]
Y_test_downsampled = downsampled_test['lesividad']


# In[ ]:


X_test_downsampled  = X_test_downsampled.astype(int)

X_test_downsampled = normalize_data(X_test_downsampled)


# In[ ]:


Y_test_downsampled


# ## XGBoost

# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# ### Genético

# In[ ]:


# # from sklearn.preprocessing import StandardScaler

# # sc = StandardScaler()
# # X_train = sc.fit_transform(X_train)
# # X_test  = sc.transform(X_test)

# # XGboost Classifier
# # model xgboost
# # use xgboost API now

# import xgboost as xgb
# import random

# Y_train_onehot = casualty_to_one_hot(Y_train)
# Y_test_onehot  = casualty_to_one_hot(Y_test_downsampled)

# numberOfParents = 40 # number of parents to start
# numberOfParentsMating = 20 # Number of parents that will mate
# numberOfParameters = 9  # Number of parameters that will be optimized
# numberOfGenerations = 100 # Number of genration that will be created 

# # Define the population size
# populationSize = (numberOfParents, numberOfParameters) # initialize the population with randomly generated parameters

# population = initilialize_poplulation(numberOfParents) # Define an array to store the fitness  hitory
# fitnessHistory = np.empty([numberOfGenerations+1, numberOfParents]) # Define an array to store the value of each parameter for each parent and generation
# populationHistory = np.empty([(numberOfGenerations+1)*numberOfParents, numberOfParameters]) # Insert the value of initial parameters in history

# populationHistory[0:numberOfParents, :] = population

# for generation in range(numberOfGenerations):
#     print("This is number %s generation" % (generation))

#     xgbDMatrixTrain = xgb.DMatrix(data = X_train, label = Y_train)
#     xgbDMatrixTest  = xgb.DMatrix(data = X_test_downsampled, label = Y_test_downsampled)
    
#     # Train the dataset and obtain fitness
#     fitnessValue = train_population(population = population,
#                                     dMatrixTrain = xgbDMatrixTrain,
#                                     dMatrixTest =  xgbDMatrixTest,
#                                     y_test = Y_test_downsampled)

#     fitnessHistory[generation, :] = fitnessValue
    
#     # Best score in the current iteration
#     print('Best F1 score in the this iteration = {}'.format(np.max(fitnessHistory[generation, :]))) # Survival of the fittest - take the top parents, based on the fitness value and number of parents needed to be selected
    
#     parents = new_parents_selection(population = population,
#                                     fitness = fitnessValue,
#                                     numParents = numberOfParentsMating)
    
#     # Mate these parents to create children having parameters from these parents (we are using uniform crossover)
#     children = crossover_uniform(parents = parents,
#                                  childrenSize = (populationSize[0] - parents.shape[0], numberOfParameters))
    
#     # Add mutation to create genetic diversity
#     children_mutated = mutation(children, numberOfParameters)
    
#     '''
#     We will create new population, which will contain parents that where selected previously based on the
#     fitness score and rest of them  will be children
#     '''
#     population[0:parents.shape[0], :] = parents # Fittest parents
#     population[parents.shape[0]:, :] = children_mutated # Children
    
#     populationHistory[(generation+1)*numberOfParents : (generation+1)*numberOfParents+ numberOfParents , :] = population # Srore parent information
    
# #Best solution from the final iteration

# fitness = train_population(population = population,
#                            dMatrixTrain = xgbDMatrixTrain,
#                            dMatrixTest  = xgbDMatrixTest,
#                            y_test = Y_test_downsampled)

# fitnessHistory[generation+1, :] = fitness # index of the best solution
# bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]

# best_hyperparams = {}
# best_hyperparams['eta'] = population[bestFitnessIndex][0]
# best_hyperparams['n_estimators']  = population[bestFitnessIndex][1]
# best_hyperparams['max_depth'] = population[bestFitnessIndex][2]
# best_hyperparams['min_child_weight'] = population[bestFitnessIndex][3]
# best_hyperparams['gamma'] = population[bestFitnessIndex][4]
# best_hyperparams['subsample'] =  population[bestFitnessIndex][5]
# best_hyperparams['colsample_bytree'] =  population[bestFitnessIndex][6]
# best_hyperparams['reg_alpha'] =  population[bestFitnessIndex][7]
# best_hyperparams['reg_lambda'] =  population[bestFitnessIndex][8]

# x_fitness = [np.max(fitnessHistory[i]) for i in range(0,fitnessHistory.shape[0])]

# FILE_NAME = 'madrid_ga_' + MODEL_TIMESTAMP  + '.jpg'

# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(len(x_fitness)), x_fitness)
# plt.savefig(GA_SCORES_PATH + FILE_NAME)


# ### Hiperparámetros

# - [Bayesian Optimization with HYPEROPT](https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook)

# #### Carga hiperparámetros

# In[ ]:


# FILE_NAME = 'madrid_hyperparams_v7.json'

# best_hyperparams = load_json(HYPERPARAMS_PATH, FILE_NAME)


# #### Cálculo de Hiperparámetros

# In[ ]:


# Y_train_onehot = casualty_to_one_hot(Y_train)
# Y_test_onehot  = casualty_to_one_hot(Y_test)

# space={'max_depth': hp.quniform("max_depth", 3, 20, 1),
#         'gamma': hp.uniform ('gamma', 1,8),
#         'reg_alpha' : hp.quniform('reg_alpha', 40, 150, 1),
#         'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#         'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#         'min_child_weight' : hp.quniform('min_child_weight', 0, 15, 1),
#         'n_estimators': hp.quniform('n_estimators', 100, 200, 100),
#         'tree_method': 'gpu_hist'
#     }

# def objective(space):
#     clf = XGBClassifier(n_estimators = int(space['n_estimators']),
#                         max_depth = int(space['max_depth']),
#                         gamma = space['gamma'],
#                         reg_alpha = int(space['reg_alpha']),
#                         min_child_weight = int(space['min_child_weight']),
#                         colsample_bytree = int(space['colsample_bytree']),
#                         tree_method = space['tree_method']
#                        )
    
#     evaluation = [(X_train, Y_train), (X_test, Y_test)]
    
#     clf.fit(X_train, Y_train,
#             eval_set = evaluation, eval_metric = "auc",
#             early_stopping_rounds = 10, verbose = False)
            
    
#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(Y_test, pred>0.5)
#     print ("SCORE:", accuracy)
#     return {'loss': -accuracy, 'status': STATUS_OK }


# trials = Trials()

# best_hyperparams = fmin(fn = objective,
#                         space = space,
#                         algo = tpe.suggest,
#                         max_evals = 200,
#                         trials = trials)


# #### Escritura hiperparámetros

# In[ ]:


FILE_NAME = 'madrid_hyperparams' + MODEL_TIMESTAMP + '.json'

write_json(best_hyperparams, HYPERPARAMS_PATH, FILE_NAME)
print(best_hyperparams)


# ### Pesos de características

# #### Carga definitiva/auxiliar de pesos

# In[ ]:


# FILE_NAME = 'madrid_adapted_leeds_default_weights.json'
FILE_NAME = 'madrid_weights_v7.json'
# FILE_NAME = 'madrid_weights2022-04-14-11:16:13.json'

feature_vector = load_json(WEIGHTS_PATH, FILE_NAME)
display(feature_vector)


# #### Cálculo de pesos de caracetrísticas

# In[ ]:


xgboost = XGBClassifier(best_hyperparams, tree_method = 'gpu_hist')

xgboost = XGBClassifier()
xgboost.fit(X_train, Y_train)

child_weights  = np.array(xgboost.feature_importances_)
feature_vector = fill_feature_vector(X_train, child_weights)


# #### Visualización pesos calculados

# In[ ]:


FILE_NAME = 'madrid_figure_weights' + MODEL_TIMESTAMP + '.jpg'

print(xgboost.get_booster().get_score(importance_type= 'weight'))
plt.figure(figsize=(10, 5))
plt.barh(X_train.columns, xgboost.feature_importances_)
plt.savefig(WEIGHTS_PATH + FILE_NAME)

print(xgboost.feature_importances_)

for column, weight in zip(X_train.columns,xgboost.feature_importances_):
  print(column, weight)

child_weights  = np.array(xgboost.feature_importances_)
feature_vector = fill_feature_vector(X_train, child_weights)


# #### Escritura de pesos de características

# - v5: Pesos calculados con hiperparámetros. En el dataset están tipificados los vehículos como en el artículo, las edades no están en rango.
# - v6: Pesos calculados con hiperparámetros. En el dataset están tipificados los vehículos como en el artículo, las edades están en rango.
# - v7: hiperparams, tipos de carretera tipificados por vía.

# In[ ]:


matrix_indexes = fv2gi(feature_vector)

FILE_NAME = 'madrid_weights' + MODEL_TIMESTAMP + '.json'
# FILE_NAME = 'default_calculated_weights.json'

write_json(feature_vector, WEIGHTS_PATH, FILE_NAME)


# ### Cálculo índices de matriz

# In[ ]:


matrix_indexes = fv2gi(feature_vector)


# ## Construcción de imágenes

# In[ ]:


train_bgi = build_gray_images(X_train, 5, matrix_indexes)
test_bgi  = build_gray_images(X_test, 5, matrix_indexes)

pd.DataFrame(train_bgi[:,:,1057])


# ## Reshape de imágenes

# In[ ]:


train_images = shape_images(X_data = X_train,
                            gray_images = train_bgi)
test_images  = shape_images(X_data = X_test,
                            gray_images = test_bgi)

plt.gray()
for i in range(0,3):
    plt.figure(figsize=(3, 3))
    plt.imshow(train_bgi[:,:,i])
    plt.show()


# In[ ]:


# train_images = shape_images(X_data = X_train,
#                             gray_images = train_bgi)
# test_images  = shape_images(X_data = X_test,
#                             gray_images = test_bgi)


# ## One-Hot

# In[ ]:


Y_train_onehot = casualty_to_one_hot(Y_train)
Y_test_onehot  = casualty_to_one_hot(Y_test)

array_train_images = np.asarray(train_images)
array_test_images  = np.asarray(test_images)


# ## Visualización de datos

# In[ ]:


# !conda install -c anaconda seaborn --y


# ### Matriz de correlación

# In[ ]:


# correlation_matrix(X_test)


# ### PCA

# In[ ]:


# pca(X_train, X_test)


# ### TSNE

# In[ ]:


# output_file_name = './2d_test_tsne.jpg'
# plot_TSNE(X_test, Y_test, n_components=2)

# output_file_name = './2d_train_tsne.jpg'
# plot_TSNE(X_train, Y_train, n_components=2)

# output_file_name = './3d_test_tsne.jpg'
# plot_TSNE(X_test, Y_test, n_components=3)

# output_file_name = './3d_train_tsne.jpg'
# plot_TSNE(X_train, Y_train, n_components=3)


# ### Autoencoder

# #### Entrenamiento

# In[ ]:


# input_img = Input(shape=(25,))

# # definimos el encoder, que tendra una entrada de Input_img y una segunda capa con entrada de encoder1 y salida 3
# encoder1 = layers.Dense(15, activation='sigmoid')(input_img)
# encoder2 = layers.Dense(3, activation='sigmoid')(encoder1)

# # definimos el  decoder que tendra una entrada inicial de encoder3 y una salida de 128 y finalmete una capa de salida con los mismos que Input_img
# decoder1 = layers.Dense(15, activation='sigmoid')(encoder2)
# decoder2 = layers.Dense(25, activation='sigmoid')(decoder1)

# # this model maps an input to its reconstruction
# autoencoder = tf.keras.Model(inputs=input_img, outputs=decoder2)
# autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy') #se usan estos dos en estas arquitecturas

X_train = array_train_images
X_test = array_test_images

X_train = X_train.reshape(len(array_train_images), 25)
X_test  = X_test.reshape(len(X_test), 25)

# autoencoder.fit(X_train, X_train,
#                 epochs=15,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_data=(X_test, X_test))


# #### Visualización

# In[ ]:


# # create encoder model
# encoder = tf.keras.Model(inputs=input_img, outputs=encoder2)
# encoder.summary()
# # create decoder model
# encoded_input = Input(shape=(3,))
# #lo que hace aqui es quedarse con las capas que corresponden al decodificador
# decoder_layer1 = autoencoder.layers[-2]
# decoder_layer2 = autoencoder.layers[-1]
# decoder = tf.keras.Model(inputs=encoded_input, outputs=decoder_layer2(decoder_layer1(encoded_input)))
# decoder.summary()
# # si miramos la salida, son simetricos el uno respecto al otro
# # encoder va de input a 3 y decoder de 3 a input

# # get latent vector for visualization
# latent_vector = encoder.predict(X_test)
# # get decoder output to visualize reconstructed image
# reconstructed_imgs = decoder.predict(latent_vector)


# # visualize in 3D plot
# from pylab import rcParams
# from mpl_toolkits.mplot3d import Axes3D

# rcParams['figure.figsize'] = 10, 8

# fig = plt.figure(1)
# ax = Axes3D(fig)

# xs = latent_vector[:, 0]
# ys = latent_vector[:, 1]
# zs = latent_vector[:, 2]

# # color=['red','green','blue']

# # for x, y, z, label in zip(xs, ys, zs, Y_test):
# #     c = color[int(label)]
# #     ax.text(x, y, z, label, backgroundcolor=c)
    
# # ax.set_xlim(xs.min(), xs.max())
# # ax.set_ylim(ys.min(), ys.max())
# # ax.set_zlim(zs.min(), zs.max())

# # plt.show()

# # X_test_encoded = encoder.predict(X_test, batch_size=32)
# # plt.figure(figsize=(6, 6))
# # plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], c=Y_test)
# # plt.colorbar()
# # plt.show()


# ## TASP-CNN

# ### Entrenamiento

# In[ ]:


array_train_images = np.asarray(train_images)
array_test_images  = np.asarray(test_images)

display(array_train_images.shape)
display(Y_train_onehot.shape)

display(array_test_images.shape)
display(Y_test_onehot.shape)

input_train_shape = (len(array_train_images), 5, 5, 1)
input_test_shape  = (len(array_test_images), 5, 5, 1)

array_train_images = array_train_images.reshape(input_train_shape)
array_test_images  = array_test_images.reshape(input_test_shape)


history = tasp_cnn.fit(array_train_images, Y_train_onehot,
                       batch_size = 128, epochs = 100, shuffle = True,
                       validation_data = (array_test_images, Y_test_onehot))

# device = cuda.get_current_device()
# device.reset()


# ### Escritura del modelo

# In[ ]:


tasp_cnn.save(MODELS_PATH + 'madrid_model_bayesian' + MODEL_TIMESTAMP + '.h5')


# ### Carga de modelo pre-entrenado

# In[ ]:


# tasp_cnn = tf.keras.models.load_model(root_path + 'madrid_model_bayesian' + MODEL_TIMESTAMP + '.h5')


# ### Resultados

# In[ ]:


from sklearn.metrics import classification_report

Y_test_labels = one_hot_to_casualty(Y_test)

########################################################################

F1_SCORE_PATH = 'F1scores/'
F1_SCORE_NAME = 'madrid_f1_score' + MODEL_TIMESTAMP

## Plot history: F1 SCORE
figure_name = plt.figure(figsize=(20, 10))
plt.plot(history.history['f1_score'], label='F1 score (training data)')
plt.plot(history.history['val_f1_score'], label='F1 score (validation data)')
plt.title('F1 score')
plt.ylabel('F1 score value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig(F1_SCORE_PATH + F1_SCORE_NAME + '.jpg')
plt.show()

print(history)

########################################################################

# evaluate the network
print("[INFO] evaluating network...")
predictions = tasp_cnn.predict(x=array_test_images, batch_size=128)

report = classification_report(tf.argmax(Y_test_onehot, axis=1),
                               predictions.argmax(axis=1),
                               target_names = Y_test_labels.unique(),
                               output_dict = True)

REPORTS_PATH = 'Reports/'
REPORT_NAME  = 'madrid_report' + MODEL_TIMESTAMP + '.csv'

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(REPORTS_PATH + REPORT_NAME, index= True)

report_df


# In[ ]:


# tasp_cnn.save(root_path + 'madrid_model_XGBOOST_predicted.h5')


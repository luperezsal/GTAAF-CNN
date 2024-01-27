import pandas as pd
from os.path import exists
import seaborn as sns
import matplotlib.pyplot as plt 
import geopandas as gpd
import contextily as cx
import matplotlib.colors as colors
import os

def plot_accidents_in_map(data_frame, severity_column, latitude_name, longitude_name, bounds):
    xmin, xmax, ymin, ymax = bounds

    pad = 5

    data_frame_2 = data_frame.replace({'Slight': 'Non-Assistance'})
    gdf = gpd.GeoDataFrame(data_frame_2,
                           geometry = gpd.points_from_xy(data_frame_2[longitude_name], data_frame_2[latitude_name]),
                           crs = "EPSG:4326")

    df_wm = gdf.to_crs(epsg=4326)
    ax = df_wm.plot(figsize=(20, 20), column=severity_column, edgecolor="k", legend=True)

    ax.set_xlim(xmin-pad, xmax+pad)
    ax.set_ylim(ymin-pad, ymax+pad)

    cx.add_basemap(ax)
    return cx


def madrid_plot_accidents_in_map(data_frame, severity_column, latitude_name, longitude_name, provider, name, city_name, bounds):
    xmin, ymin, xmax, ymax  = bounds

    pad = 0

    data_frame_2 = data_frame.replace({'Slight': 'Non-Assistance'})

    gdf = gpd.GeoDataFrame(data_frame_2,
                           geometry = gpd.points_from_xy(data_frame_2[longitude_name], data_frame_2[latitude_name]),
                           crs = "EPSG:4326")

    os.makedirs(f"providers/{city_name}", exist_ok=True)

    gdf = gdf.cx[xmin:xmax, ymin:ymax]
    df_wm = gdf.to_crs(epsg=3857)

    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot()

    df_wm.plot(column = severity_column,
    		   edgecolor = "k",
    		   legend = True,
    		   ax = ax,
    		   markersize=15,
    		   cmap=colors.ListedColormap(['#D94325','#5CD925']))
    #ax.set_xlim(xmin-pad, xmax+pad)
    #ax.set_ylim(ymin-pad, ymax+pad)
    cx.add_basemap(ax, source=provider)

    fig.savefig(f"providers/{city_name}/{name}_{str(provider['name'])}.png", bbox_inches="tight")

	
    return



def make_reports_summary(times, city_name, MODEL_TIMESTAMP, REPORTS_PATH, REPORTS_SUMMARY_PATH, leeds, madrid, UK):
    reports_summary = pd.DataFrame()

    cities = []
    # MODEL_TIMESTAMP = '2022-08-02-10:10:19'

    cities.append('leeds')  if leeds else None
    cities.append('madrid') if madrid else None
    cities.append(f'{city_name}') if UK else None

    models_renaming = {'knn': 'KNN',
                       'convolution_1d': '1D-convolution',
                       'convolution_2d': '2D-convolution',
                       'nb': 'NB',
                       'svc': 'SVC',
                       'logistic_regression': 'Logistic-Regression',
                       'random_forest': 'Random-Forest',
                       'mlp': 'ML-Perceptron',                   
                      }
                       # 'auto_ml': 'AutoML'}

    splits = ['train', 'test']
    sorted_by_time_models_name = times.model

    for split in splits:
        reports_summary = pd.DataFrame()

        for model_name in sorted_by_time_models_name:

            REPORT_PATH = f"{REPORTS_PATH}{model_name}/{split}/"

            for city_name in cities:

                REPORT_NAME  = f"{city_name}_{model_name}_report_{MODEL_TIMESTAMP}.csv"

                if exists(REPORT_PATH + REPORT_NAME):
                    print(f"Found: {model_name} for {split}")
                    report = pd.read_csv(REPORT_PATH + REPORT_NAME, index_col=[0])
                    report.insert(0, 'split', split)
                    report.insert(1, 'city', city_name)
                    report.insert(2, 'model', models_renaming[model_name])

                    reports_summary = pd.concat([reports_summary, report])

                    reports_summary = reports_summary.sort_values(['city', 'model'], ascending = [True, True])

        if not reports_summary.empty:
            c_m = reports_summary['city'] + '_' + reports_summary['model']
            reports_summary.insert(0, 'c_m', c_m)

            SAVE_PATH =  f"{REPORTS_SUMMARY_PATH}/{split}/{MODEL_TIMESTAMP}.csv"

            reports_summary.insert(0, 'accident_type', reports_summary.index)
            reports_summary.to_csv(SAVE_PATH, index= True)

    return reports_summary, splits, cities



def make_f1_score_barplot(reports_summary, splits, city_name, cities, MODEL_TIMESTAMP, REPORTS_SUMMARY_PATH, leeds, madrid, UK):
	MEASURE_TYPES  = ['precision', 'recall', 'f1-score']
	# ACCIDENT_TYPES = ['Slight', 'Serious', 'Fatal']

	ACCIDENT_TYPES = ['Slight', 'Assistance']

	sns.set_style("whitegrid")

	if leeds:
	    leeds_reports_summary  = reports_summary[reports_summary['city'] == 'leeds']
	if False: #madrid:
	    madrid_reports_summary = reports_summary[reports_summary['city'] == 'madrid']
	if UK:
	    UK_reports_summary = reports_summary[reports_summary['city'] == city_name]

	# print(leeds_reports_summary.loc[ACCIDENT_TYPES])

	for split in splits:
	    
	    REPORT_PATH = f"{REPORTS_SUMMARY_PATH}{split}/{MODEL_TIMESTAMP}.csv"

	    if exists(REPORT_PATH):
	        fig, axs = plt.subplots(len(MEASURE_TYPES), len(cities), figsize=(15,20))
	        plt.subplots_adjust(bottom=0.05, top=0.97)

	        print(f"Found: {REPORT_PATH}")

	        report = pd.read_csv(REPORT_PATH, index_col=[0])

	        if leeds:
	            leeds_reports_summary  = report[report['city'] == 'leeds']
	        if madrid:
	            madrid_reports_summary = report[report['city'] == 'madrid']
	        if UK:
	            UK_reports_summary = report[report['city'] == city_name]

	        for index, measure_type in enumerate(MEASURE_TYPES):

	            capitalized_measure_type = measure_type.capitalize()

	            # Si son dos ciudades el plot es bidimensional.
	            if len(cities) > 1:
	                axis_leeds = axs[index, 0]
	                axis_madrid = axs[index, 1]
	            else:
	                axis_leeds = axis_madrid = axis_UK = axs[index]

	            if leeds:
	                ax = sns.barplot(x = 'accident_type',
	                                 y = measure_type,
	                                 hue = 'model',
	                                 palette = 'deep',
	                                 data = leeds_reports_summary.loc[ACCIDENT_TYPES],
	                                 ax = axis_leeds).set(title = f"{measure_type} Leeds")


	            if madrid:
	                ax = sns.barplot(x = 'accident_type',
	                                 y = measure_type,
	                                 hue = 'model',
	                                 palette = 'deep',
	                                 data = madrid_reports_summary.loc[ACCIDENT_TYPES],
	                                 ax = axis_madrid)
	            
	            if UK:
	                ax = sns.barplot(x = 'accident_type',
	                                 y = measure_type,
	                                 hue = 'model',
	                                 palette = 'deep',
	                                 data = UK_reports_summary.loc[ACCIDENT_TYPES],
	                                 ax = axis_UK)

	            ax.set_ylabel(capitalized_measure_type, fontsize=18)
	            ax.set(title = f"{capitalized_measure_type} {city_name}")
	            ax.legend(loc='lower center', fontsize=13)

	        SAVE_PATH = f"{REPORTS_SUMMARY_PATH}{split}/{MODEL_TIMESTAMP}.png"

	        fig = fig.get_figure()
	        
	        fig.savefig(SAVE_PATH,  dpi=400)
	plt.show()


def plot_time_series():

	plt.rcParams.update({
	    "text.usetex": True,
	})

	root_path = 'Reports/summary/test/'

	all_cities_summaries = pd.DataFrame()

	best_results_mapper = {'Birmingham': '2023-08-18-15:47:06',
	                       'Sheffield':  '2023-08-18-17:08:06',
	                       'Liverpool':  '2023-08-18-15:25:24',
	                       'Southwark':  '2023-08-18-18:08:44',
	                       'Manchester': '2023-08-18-17:44:54',
	                       'Cornwall':   '2023-10-13-13:21:02',
	                       'Victoria':   '2023-10-10-00:15:31',
	                       'Madrid':     '2023-10-10-13:04:10'
	                       }

	for city_name, city_timestamp in best_results_mapper.items():
	    city_summary = pd.read_csv(f"{root_path}/{city_timestamp}.csv", index_col=0)

	    # City names Capitalized
	    city_summary['city'] = city_summary['city'].apply(lambda x: x.capitalize())

	    all_cities_summaries = pd.concat([all_cities_summaries, city_summary])

	all_cities_summaries = all_cities_summaries.drop_duplicates()

	casualty_types = ['Assistance']

	all_cities_summaries = all_cities_summaries[all_cities_summaries['model'] != '1D-convolution']
	all_cities_summaries.index = all_cities_summaries.city

	for casualty_type in casualty_types:
	    current_casualty_type_all_cities_summaries = all_cities_summaries[all_cities_summaries['accident_type'] == casualty_type]


	    data_grouped_by_model = current_casualty_type_all_cities_summaries.groupby('model')['f1-score']

	    for model_name, cities_model_metrics in data_grouped_by_model:

	        # rc('text', usetex=True)

	        if model_name == '2D-convolution':
	            cities_model_metrics.plot(x = 'city',
	            						  figsize = (20, 10),
	            						  grid = True,
	            						  label = r"\textbf{OURS}",
	            						  linewidth = 3,fontsize=15)
	        else:
	            cities_model_metrics.plot(x = 'city',
	            						  figsize = (20, 10),
	            						  grid = True,
	            						  label = f"{model_name}",
	            						  linestyle = 'dashed')

	    plt.legend(fontsize=14)
	    plt.title(label = f'Models F1-scores by city ({casualty_type} Accidents)', fontsize=15)
	    plt.xlabel('City', fontsize=18)
	    plt.ylabel('F1-Score', fontsize=18)
	    plt.savefig(f"{casualty_type}_a.svg")





import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def plot_radial_graph():


	plt.rcParams.update({
	    "text.usetex": True,
	})

	root_path = 'Reports/summary/test/'

	all_cities_summaries = pd.DataFrame()

	best_results_mapper = {'Birmingham': '2023-08-18-15:47:06',
	                       'Sheffield':  '2023-08-18-17:08:06',
	                       'Liverpool':  '2023-08-18-15:25:24',
	                       'Southwark':  '2023-08-18-18:08:44',
	                       'Manchester': '2023-08-18-17:44:54',
	                       'Cornwall':   '2023-07-15-15:49:36',
	                       'Victoria':   '2023-10-10-00:15:31',
	                       'Madrid':     '2023-10-10-13:04:10'
	                       }

	for city_name, city_timestamp in best_results_mapper.items():
	    city_summary = pd.read_csv(f"{root_path}/{city_timestamp}.csv", index_col=0)

	    # City names Capitalized
	    city_summary['city'] = city_summary['city'].apply(lambda x: x.capitalize())

	    all_cities_summaries = pd.concat([all_cities_summaries, city_summary])

	all_cities_summaries = all_cities_summaries.drop_duplicates()

	casualty_types = ['Slight', 'Assistance']

	all_cities_summaries = all_cities_summaries[all_cities_summaries['model'] != '1D-convolution']
	all_cities_summaries.index = all_cities_summaries.city


	for casualty_type in casualty_types:
	    fig = go.Figure()

	    current_casualty_type_all_cities_summaries = all_cities_summaries[all_cities_summaries['accident_type'] == casualty_type]

	    data_grouped_by_model = current_casualty_type_all_cities_summaries.groupby('model')['f1-score']

	    graph_type = 'hist'

	    for model_name, cities_model_metrics in data_grouped_by_model:

	        # rc('text', usetex=True)

	        a = pd.DataFrame(cities_model_metrics)
	        a['city'] = a.index
	        a.reset_index(drop=True, inplace=True)

	        if model_name == '2D-convolution':
	            fig.add_trace(go.Scatterpolar(
		              r=np.append(a['f1-score'].values, a['f1-score'].values[0]),
		              theta=np.append(a['city'].values, a['city'].values[0]),
	                  name='<b>OURS</b>',
	                  opacity=1,
	                  mode='lines+markers',
	            ))
	        else:
		        fig.add_trace(go.Scatterpolar(
		              r=np.append(a['f1-score'].values, a['f1-score'].values[0]),
		              theta=np.append(a['city'].values, a['city'].values[0]),
		              fill=None,
		              name=model_name,
		              opacity=0.4,
		              mode='lines+markers',
		        ))


	    if casualty_type == 'Slight':
	    	casualty_range = [0.2, 1]
	    else:
	    	casualty_range = [0.3, 1]

	    fig.update_layout(
	    title = dict(text=f"F1-Score by city ({casualty_type} accidents)", font=dict(size=20), yref='paper'),
	    polar = dict(
	        radialaxis=dict(
	          visible=True,
	          range=casualty_range
	        )),
	    showlegend=True
	)

	    fig.show()
	    fig.write_image(f"{casualty_type}.svg", format='svg')




def plot_victoria_sin_cos_hours_in_circle(data_frame, save_path):
    plt.rcParams.update({'font.size': 15})
    
    index_of_selected_hours = [6917, 8469, 4190, 841, 4259, 4351, 1175, 4840, 3058, 9299, 6383, 4689, 6512, 6564, 7750, 4608]

    sin = np.asarray(data_frame['Accident Time Cos'].loc[index_of_selected_hours])
    cos = np.asarray(data_frame['Accident Time Sin'].loc[index_of_selected_hours])
    labels = data_frame['Accident Time'].loc[index_of_selected_hours]
    labels = labels.apply(lambda x: x.replace('.', ':'))

    # Set Seaborn style
    sns.set(style='whitegrid')

    # Plot the circle
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed

    plt.scatter(cos, sin, color='purple', label='Points', s=120, edgecolors='black', linewidth=1.5)

    for i, (x, y, label) in enumerate(zip(cos, sin, labels)):
        plt.text(x + 0.05, y, label, fontsize=10, ha='left', va='center')

    # Add labels and title
    plt.xlabel('Cosine Value', fontsize=15)
    plt.ylabel('Sine Value', fontsize=15)
    plt.title('Plot of Sin/Cos Accidents Hour', fontsize=18)

    # Set font size for tick labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Increment bounding axis for extra space
    buffer = 0.9
    plt.xlim(buffer)
    plt.ylim(buffer)

    # Equal aspect ratio to make it a perfect circle
    plt.axis('equal')

    # Show the plot
    plt.savefig(save_path)
    plt.show()

def plot_victoria_sin_cos_hours_in_circle_2(data_frame, save_path):
	plt.rcParams.update({'font.size': 15})

	index_of_selected_hours = [6917, 8469, 4190, 841, 4259, 4351, 1175, 4840, 3058, 9299, 6383, 4689, 6512, 6564, 7750, 4608]

	sin = np.asarray(data_frame['Accident Time Cos'].loc[index_of_selected_hours])
	cos = np.asarray(data_frame['Accident Time Sin'].loc[index_of_selected_hours])
	labels = data_frame['Accident Time'].loc[index_of_selected_hours]
	labels = labels.apply(lambda x: x.replace('.', ':'))

	# Set Seaborn style
	sns.set(style='whitegrid')

	# Convert angles from radians to degrees
	angles = np.arctan2(sin, cos) * 180 / np.pi

	# Plot the polar plot
	plt.figure(figsize=(10, 10))
	ax = plt.subplot(111, projection='polar')

	# Plot points with colors based on hours
	scatter = ax.scatter(np.radians(angles), np.sqrt(sin**2 + cos**2), c=np.arange(len(labels)), cmap='viridis', s=120, edgecolors='black', linewidth=1.5)

	# Add labels
	for i, (angle, label) in enumerate(zip(np.radians(angles), labels)):
	    ax.text(angle, 1.1, label, ha='left', va='center', fontsize=10)

	# Add color bar
	cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
	cbar.set_label('Hour Index', rotation=270, labelpad=15)

	# Set title
	plt.title('Polar Plot of Sin/Cos Accidents Hour', fontsize=18)

	# Show the plot

	plt.savefig(save_path)
	plt.show()

	# Example usage
	# plot_victoria_sin_cos_hours_in_circle(your_data_frame)
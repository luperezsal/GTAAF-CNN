import pandas as pd
from os.path import exists
import seaborn as sns
import matplotlib.pyplot as plt 


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
	                                 ax = axis_madrid).set(title = f"{measure_type} Madrid")
	            
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
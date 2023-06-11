from src.data import dayNameFromWeekday
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import os


def plot_week_days(new_dataframe, name, filters=[]):

    ec = {'Fatal': 'red',
          'Serious': 'yellow',
          'Slight': 'green'}

    for week_day in new_dataframe['dia_semana'].unique():
        week_day_df = new_dataframe[new_dataframe['dia_semana'] == week_day]

        if filters:
            week_day_df = week_day_df[week_day_df.tipo_vehiculo.isin(filters)]
        week_day_df['color'] = week_day_df.lesividad.apply(lambda x: ec[x])

        week_day_gpd = gpd.GeoDataFrame(week_day_df,
                                        geometry = gpd.points_from_xy(week_day_df.Longitude, week_day_df.Latitude),
                                        crs = "EPSG:4326")
       
        # week_day_gpd.total_bounds = gdf.total_bounds

        week_day_gpd = week_day_gpd.to_crs(epsg=3857)
        ax = week_day_gpd.plot(figsize=(20, 20), column='lesividad', legend=True, color=week_day_gpd['color'])
        cx.add_basemap(ax)

        store_path = f'maps/Madrid/week_days/{name}'

        os.makedirs(store_path, exist_ok=True)

        image_name = f"madrid_{week_day}_{dayNameFromWeekday(week_day)}.png"

        plt.savefig(
            f"{store_path}/{image_name}",
            dpi=200,
            bbox_inches="tight",
            title = 'a'
        )
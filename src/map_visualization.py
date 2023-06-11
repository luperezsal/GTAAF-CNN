import geopandas as gdp
import matplotlib.pyplot  as plt
import numpy as np
import shapely

def build_grid(gdf, x_offset, y_offset):
	# Hacer esto con el dataframe utm original
	# 

	# total area for the grid
	xmin, ymin, xmax, ymax= gdf.total_bounds

	# # how many cells across and down
	# 
	# n_cells = 30
	# cell_size = (xmax-xmin)/n_cells

	# projection of the grid
	crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"

	# create the cells in a loop
	grid_cells = []

	for x0 in np.arange(xmin, xmax+x_offset, x_offset):

	    for y0 in np.arange(ymin, ymax+y_offset, y_offset):

	        # bounds
	        x1 = x0-x_offset
	        y1 = y0+y_offset

	        grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

	cell = gdp.GeoDataFrame(grid_cells,
							columns = ['geometry'], 
	                        crs = crs)

	return cell

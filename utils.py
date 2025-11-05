import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from xskillscore import crps_ensemble
import os
import random
import torch

def get_region(name, regions):
    for region in regions:
        if region[0] == name:
            return region
    return None 

def create_multi_plot(
            data,
            vmin, vmax,
            fig_path, fig_name,
            n_rows, n_cols,
            cmap_min=None, cmap_max=None,
            cmap_colors=((0, 1, 10)),
            cmap_first_color=[0], cmap_last_color=[None],             
            color='cool',
            rtick_bool=[None], ltick_bool=[None],
            orientation='horizontal', spacing='uniform',
            var='tasmean', fontsize=16, y_height = 0.02,
            title=None, x_map=None,
            y_map=None, y_map_pos=(-0.07, 0.55)
    ):
        rtick_bool = rtick_bool*n_rows if rtick_bool==[None] else rtick_bool
        ltick_bool = ltick_bool*n_rows if ltick_bool==[None] else ltick_bool
        cmap_last_color = cmap_last_color*n_rows if cmap_last_color==[None] else cmap_last_color
        color = color if isinstance(color, list) else [color]*n_rows

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
        for i, (predictand_name, predictand_data) in enumerate(data.items()): 
            for j, (metric, metric_data) in enumerate(predictand_data.items()): #metric
                #Color map for Row 
                continuousCMAP = plt.get_cmap(color[j])
                discreteCMAP = ListedColormap(continuousCMAP(np.linspace(*cmap_colors[j])[cmap_first_color[j]:cmap_last_color[j]]))

                if n_cols == 1 and n_rows == 1:
                    ax = axes
                elif n_cols == 1:
                    ax = axes[j]
                elif n_rows == 1:
                    ax = axes[i]
                else:
                    ax = axes[j, i]

                if j == 0 and (title is not None or x_map is not None):
                    ax.set_title(title if title!=None else f'{x_map[predictand_name]}', fontsize=fontsize)
                if i == 0 and y_map is not None:
                    ax.text(y_map_pos[0], y_map_pos[1], f'{y_map[metric]}', va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor',
                        transform=ax.transAxes, fontsize=fontsize)
                ax.coastlines(resolution='10m')            

                dataToPlot = metric_data if var is None else metric_data[var]
                im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
                                    dataToPlot,
                                    transform=ccrs.PlateCarree(),
                                    cmap=discreteCMAP,
                                    vmin=vmin[j], vmax=vmax[j])
                
                mean_val = float(np.nanmean(dataToPlot))
                min_val = float(np.nanmin(dataToPlot))
                max_val = float(np.nanmax(dataToPlot))

                ax.text(
                    0.97-0.35, 0.03,
                    f"{mean_val:.2f}",
                    transform=ax.transAxes,
                    fontsize=fontsize * 0.6,
                    color="black",
                    ha="right", va="bottom"
                )
                ax.text(
                    0.97-0.15, 0.03, 
                    f"({min_val:.2f} - ",
                    transform=ax.transAxes,
                    fontsize=fontsize * 0.6,
                    color="#1f77b4",
                    ha="right", va="bottom"
                )
                ax.text(
                    0.97, 0.03,
                    f"{max_val:.2f})",
                    transform=ax.transAxes,
                    fontsize=fontsize * 0.6,
                    color="#d62728",
                    ha="right", va="bottom"
                )

                if i == 0:
                    if n_rows >= 3:
                        y_pos = 0.655 - (j * 0.305)
                    elif n_rows == 2:
                        y_pos = 0.510 - (j * 0.443)
                    else:
                        y_pos = 0.08
                    position = [0.125, y_pos, 0.775, y_height]
                    _add_colorbar(fig=fig, im=im, position=position,
                            vmin=vmin[j],
                            vmax=vmax[j],
                            cmap_min=cmap_min[j],
                            cmap_max=cmap_max[j],
                            rtick_bool=rtick_bool[j], 
                            ltick_bool = ltick_bool[j],
                            orientation=orientation,
                            spacing=spacing,
                            fontsize=fontsize)

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{fig_path}/{fig_name}', bbox_inches='tight')
        plt.close()

def _add_colorbar(fig, im, position, vmin, vmax, 
                ltick_bool, rtick_bool,
                cmap_min=0, cmap_max=5, 
                fontsize=16,
                spacing='uniform', orientation=None):
        cax = fig.add_axes(position)
        cbar = plt.colorbar(im, cax, orientation=orientation, spacing=spacing)
        ticks = np.linspace(vmin, vmax, int(np.floor(cmap_max - cmap_min)))
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=fontsize)
        if rtick_bool or ltick_bool:
            tick_labels = [tick.get_text() for tick in cbar.ax.get_xticklabels()]
            if rtick_bool:
                tick_labels[-1] += '+' 
            if ltick_bool:
                tick_labels[0] += '-' 
            cbar.ax.set_xticklabels(tick_labels)

def get_predictand(data_path, name, var, complete_path = None):
    if complete_path == None:
        file_name = get_file_name(data_path, name, keyword = var)
        predictand_path = f'{data_path}{name}/{file_name}'
    else:
        predictand_path = complete_path
    print(f"PREDICTAND PATH: {predictand_path}")
    predictand = xr.open_dataset(predictand_path,
                                chunks=-1, mode='r') # Near surface air temperature (daily mean)
    predictand = _check_correct_data(predictand) # Transform coordinates and dimensions if necessary

    predictand = _check_index(predictand)
    predictand = _check_units_tempt(predictand, var)
    predictand = _remove_wrong_data(predictand, var, name)
    predictand = predictand.assign_coords({'time': predictand.indexes['time'].normalize()})

    return predictand

def mask_data(var, objective, secondGrid=None, grid = None, path = None, to_slice=None):
    """_summary_

    Args:
        var (_type_): _description_
        objective (_type_): _description_
        secondGrid (_type_, optional): _description_. Defaults to None.
        grid (_type_, optional): _description_. Defaults to None.
        path (_type_, optional): _description_. Defaults to None.
        to_slice (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if path != None:
        grid = xr.open_dataset(path, mode='r')
    grid = _check_correct_data(grid) # Transform coordinates and dimensions if necessary
    grid = _check_index(grid)
    grid=grid.assign_coords({'time': grid.indexes['time'].normalize()})
    if to_slice != None:
        baseMask = flattenSpatialGrid(grid=grid.sel(time=slice(*to_slice)).load(), var=var)
    else:
        baseMask = flattenSpatialGrid(grid=grid.load(), var=var)

    objectiveFlat = baseMask.flatten(grid=objective, var=var)
    objectiveFlat_array = _to_array(objectiveFlat)
    objectiveFlat[var].values = objectiveFlat_array
    objectiveUnflatten = baseMask.unFlatten(grid=objectiveFlat, var=var)
    
    if np.isnan(objectiveUnflatten).sum() > 0 and secondGrid != None:
        secondMask = _obtain_mask(grid = secondGrid, var = var)
        secondFlat = secondMask.flatten(grid=objectiveUnflatten, var=var)
        secondFlat_array = _to_array(secondFlat)
        secondFlat[var].values = secondFlat_array
        objectiveUnflatten = secondMask.unFlatten(grid=secondFlat, var=var)

    return objectiveUnflatten


def _to_array(grid, coordNames={'lat': 'lat', 'lon': 'lon'}):
    '''
    Transforms the data from a Dataset into a numpy array, maintaining all the
    variables

    @grid: Dataset to transform
    @coordNames: Dictionary with lat and lon names of the grid
    '''

    # Get name of all the variables of the Dataset
    vars = [i for i in grid.data_vars]

    # Get name of all the dimensions of the Dataset
    dims = [i for i in grid.coords]

    # Check whether we are working with X or Y Datasets
    # In the former case we must have latitude and longitude dimensions. In the
    # latter we must have a gridpoint variable (multindex of latitude and longitude)
    if ('gridpoint' in dims) and (len(vars) == 1): # Y case

        if 'time' in grid.coords:
            npGrid = np.empty((grid['time'].shape[0], grid['gridpoint'].shape[0]))
        else:
            npGrid = np.empty((1, grid['gridpoint'].shape[0]))

        npGrid = grid[vars[0]].values

    elif (len(vars) > 1): # X case

        if 'time' in grid.coords:
            npGrid = np.empty((grid['time'].shape[0], len(vars),
                           grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))
        else:
            npGrid = np.empty((1, len(vars),
                           grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))
        
        for idx, var in enumerate(vars):
            npGrid[:, idx, :, :] = grid[var].values

    elif ('gridpoint' not in dims)  and(len(vars) == 1): # fully convolutional Y case

        if 'time' in grid.coords:
            npGrid = np.empty((grid['time'].shape[0],
                           grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))
        else:
            npGrid = np.empty((1, grid[coordNames['lat']].shape[0], grid[coordNames['lon']].shape[0]))

        npGrid = grid[vars[0]].values

    else:
        raise ValueError('Please provide a Dataset with either gridpoint or ' \
                         'latitude and longitude dimensions')

    return npGrid

def _obtain_mask(var, grid = None, path = None, to_slice=None):
    """Function that obtains a mask from dataset to apply to other datasets.

    Args:
        var (string): Variable to use as reference in dataset.
        grid (xarray.object, optional): Dataset to obtain mask from. Defaults to None.
        path (string, optional): Path to load dataset to obtain mask from. Defaults to None.
        to_slice (duple(string, string), optional): Range to slice time in case of being necessary. Defaults to None.

    Returns:
        utils.flattenSpatialGrid: An object of the class 'flattenSpatialGrid'
    """
    if path != None:
        grid = xr.open_dataset(path, mode='r')
    grid = _check_correct_data(grid) # Transform coordinates and dimensions if necessary
    grid = _check_index(grid)
    grid=grid.assign_coords({'time': grid.indexes['time'].normalize()})
    if to_slice != None:
        baseMask = flattenSpatialGrid(grid=grid.sel(time=slice(*to_slice)).load(), var=var)
    else:
        baseMask = flattenSpatialGrid(grid=grid.load(), var=var)

    return baseMask

def get_metrics_temp(data, data_reference = None, var = 'tasmean', short = False):#, mask=None):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    val_mean = data.mean(dim = 'time')
    val_mean_annual = data.resample(time = 'YE').mean()
    val_st = data.std(dim='time')
    val_st_interannual = data.groupby('time.year').std(dim = 'time').mean(dim='year')#.resample(time = 'YE')
    val_99 = data.groupby('time.year').quantile(0.99, dim = 'time').mean(dim='year')
    val_1 = data.groupby('time.year').quantile(0.01, dim='time').mean(dim='year')
    if short == False:
        over30 = data[var].where(data[var] >= 30).resample(time='YS').count(dim='time').mean(dim='time').to_dataset(name=var)
        over30 = over30.where(over30 != 0, np.nan)
        over40 = data[var].where(data[var] >= 40).resample(time='YS').count(dim='time').mean(dim='time').to_dataset(name=var)
        over40 = over40.where(over40 != 0, np.nan)
        mean_max_mean = data.resample(time = 'YE').max(dim='time').mean(dim='time')
        crps = crps_ensemble(data, data_reference) if data_reference!=None else 0

    if short:
        response = {
        'Mean': val_mean,
        '99th': val_99
        }
    else:
        response = {
        'Mean': val_mean,
        '99th': val_99,
        '1th': val_1,
        'Std': val_st,
        'Std_annual' : val_st_interannual,
        'Trend': val_mean_annual,
        'Over30': over30,
        'Over40': over40,
        'Mean_max_mean': mean_max_mean,
        'Crps': crps
        }

    return response


def get_file_name(data_path, target_name, keyword):
    """_summary_

    Args:
        data_path (_type_): _description_
        target_name (_type_): _description_
        keyword (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_name = None
    predictand_folder_path = f'{data_path}{target_name}'
    files_in_predictand_folder = os.listdir(predictand_folder_path)
    for file in files_in_predictand_folder:
        if target_name.lower() in file.lower() and keyword in file.lower():
            file_name = file
            break
    return file_name


def _check_correct_data(dataset, name_transformation={'latitude': 'lat', 'longitude': 'lon'}):
    """
    Checks dataset dimensions and coordinates names.
    If they are not in the format 'lat', 'lon', 'time', transforms them only if possible.

    Args:
        dataset (xarray.Dataset): The dataset to be checked and potentially transformed.
        name_transformation (dict, optional): A dictionary mapping old names to new names.
            Defaults to {'latitude': 'lat', 'longitude': 'lon'}.

    Returns:
        xarray.Dataset: The original dataset if no transformation is needed,
            otherwise the transformed dataset.

    Raises:
        ValueError: If transformation is not possible due to name conflicts.
    """

    # Check if transformation is needed
    current_dims = set(dataset.dims)
    current_coords = set(dataset.coords.keys())
    for key,value in name_transformation.items():

        coord_transform_req = False
        dim_transform_req = False

        if value not in current_coords:
            coord_transform_req = True

        if value not in current_dims:
            dim_transform_req = True

        if key not in current_coords:
            coord_transform_req = False

        if key not in current_dims:
            dim_transform_req = False

        # If transformation is possible, perform rename
        if dim_transform_req:
            print(f"Dimension transformation done successfully for {key}:{value}!")
            try:
                dataset = dataset.rename_dims({key: value})
            except:
                raise Exception("Dim transform error!")
        else:
            print(f"Transformation for dimensions not needed for {key}:{value}")

        if coord_transform_req:
            print(f"Coordinates transformation done successfully for {key}:{value}!")
            try:
                dataset = dataset.rename_vars({key: value})
            except:
                raise Exception("Coord transform error!")
        else:
            print(f"Transformation for coordinates not needed for {key}:{value}")

    return dataset


def _check_index(dataset):
    """_summary_

    Args:
        dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    dataset_modificado = dataset
    lat_diff = dataset['lat'].diff('lat')
    lon_diff = dataset['lon'].diff('lon')
    if np.all(lat_diff < 0):
        dataset_modificado = dataset_modificado.reindex(lat=list(reversed(dataset.lat)))
    if np.all(lon_diff < 0):
        dataset_modificado = dataset_modificado.reindex(lon=list(reversed(dataset.lon)))

    return dataset_modificado


def _check_units_tempt(data, var):
    """Check units of tempeture and transform them if necessary.

    Args:
        data (xarray.dataset): Dataset de xarray.
        var (string): Variable a transformar.

    Returns:
        xarray.dataset: Sme dataset with tranformed values if necessary.
    """
    is_kelvin = (data[var] > 100).any(dim=('time', 'lat', 'lon'))

    if is_kelvin:
        data[var].data = data[var].data - 273.15

    return data

def _remove_wrong_data(data, var='tasmean', name='CHELSA'):

    if name == 'CHELSA':
        mask = (data[var] < -200).any(dim=['lat', 'lon'])
        mask_computed = mask.compute()
        wrong_days = data['time'].where(mask_computed, drop=True)
        data_cleaned = data.drop_sel(time=wrong_days.values)
    else:
        data_cleaned = data
        
    return data_cleaned


class flattenSpatialGrid():
    '''
    Transform a 2D variable into a 1D vector. First it flattens the 2D variable,
    then it removes the observations with NANs values.
    '''

    def __init__(self, grid, var, coordNames={'lat': 'lat', 'lon': 'lon'}):
        '''
        Initialize the flattening operation wit the reference grid. This grid is
        used to compute the NAN mask.

        @grid: Reference grid containing the data to flatten. It must be a Dataset
               containing 2D variables
        @var: Variable to flatten
        '''

        self.latName = coordNames['lat']
        self.lonName = coordNames['lon']

        # Save lat and lon dimensions
        self.lat = grid[self.latName].copy()
        self.lon = grid[self.lonName].copy()

        # Save new dimensions with NANs
        newGrid = grid.stack(gridpoint=(self.latName, self.lonName))

        # Compute and save NANs mask
        self.nanIndices = np.isnan(newGrid[var])
        self.nanIndices = np.any(self.nanIndices, axis=0)
        del newGrid

        # Create refArray for getPosition functions
        self.refArray = self.nanIndices
        self.refArray = self.refArray.where(~self.nanIndices, drop=True)
        self.refArray.values = np.arange(0, self.refArray.values.shape[0])

        # Save grid's template lat
        self.gridTemplate = grid.sel(time=grid['time'].values[0])
        self.gridTemplate = self.gridTemplate.stack(gridpoint=(self.latName, self.lonName))
        self.gridTemplate = self.gridTemplate.expand_dims('time')
        self.gridTemplate = xr.where(cond=self.gridTemplate[var] != np.nan,
                                     x=np.nan, y=np.nan)

    def flatten(self, grid, var):
        '''
        Perform the flattening taking into acount the Dataset provided as reference
        in the __init__ method

        @grid: Grid containing the data to flatten. It must be a Dataset
               containing 2D variables
        @var: Variable to flatten
        '''

        # Check dimensions of grid to flatten
        if np.array_equal(grid[self.latName].values, self.lat.values) and \
           np.array_equal(grid[self.lonName].values, self.lon.values):

           # Flatten grid
           newGrid = grid.stack(gridpoint=(self.latName, self.lonName))

           # Filter NANs
           newGrid = newGrid.where(~self.nanIndices, drop=True)

           return newGrid

        else:
            raise ValueError('Discrepancies found in the latitude and longitude dimensions between grids')

    def unFlatten(self, grid, var, revertLat=False):
        '''
        Unflatten the grid taking into account the grid passed to the __init__ method

        @grid: Grid containing the data to unflatten. It must be a Dataset
               containing 2D variables
        @var: Variable to unflatten
        @var: Whether to revert the latitude coordinate
        '''

        # Create a dataset with all the gridpoints spanning the time of the input grid
        refGrid = self.gridTemplate.reindex({'time': grid['time'].values})
        
        # Merge grids and unstack
        finalGrid = grid.combine_first(refGrid)
        finalGrid = finalGrid.unstack('gridpoint')

        if revertLat:
            finalGrid = finalGrid.reindex(lat=list(reversed(finalGrid.lat)))

        return finalGrid
    


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



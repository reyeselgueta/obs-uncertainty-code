import xarray as xr
import utils
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import sys



DATA_PATH = './data/input/'
FIGS_PATH = './notebooks/figures/'
MODELS_PATH = './notebooks/models/'
PREDS_PATH = './preds/'


# INPUT DATA
FIGS = sys.argv[1]
ENSEMBLE_QUANTITY = 50
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'


# GENERAL VARIABLES
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'CHELSA']
predictands_map = {'ERA5-Land0.25deg': 'ERA5-Land', 'E-OBS': 'E-OBS','AEMET_0.25deg':'ROCIO-IBEB', 'CHELSA': 'CHELSA'}

hist_baseline = ('1995-01-01', '2014-12-31')
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
yearsShort = ('2041', '2060')
yearsMedium = ('2061', '2080')
yearsLong = ('2081', '2100')
yearsGCM = [yearsShort, yearsMedium, yearsLong]
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')

regions = [
    ("Central Plateau", 39.0, 41.5, -6.5, -2.5, "lightgrey"),# Name, Lat, Lat, Lon, Lon
    ("Cantabrian Mountains", 42.5, 43.5, -7.0, -2.5, "lightgrey"),
    ("Ebro Valley", 40.5, 42.5, -2.0, 1.5, "lightgrey"),
    ("Pyrenees", 41.42, 42.8, -0.37, 3.37, "lightgrey"),
    ("Sierra Nevada", 36.7, 37.2, -3.6, -2.8, "lightgrey"),
    ("Guadalquivir Valley", 36.8, 38.5, -6.5, -3.5, "lightgrey")
]


### # FIG1 # ####
if '1' in FIGS:
# Observational data
    metrics = ['Mean', '99th']
    metrics_map = {metric: metric for metric in metrics}
    obs = {}
    whole_obs = {'annual': {}}
    whole_obs_metrics = {'annual': {}}
    total_metrics = {f'{metric_stat}': [] for metric_stat in metrics}


    for predictand_name in predictands:

        modelName = f'DeepESD_tas_{predictand_name}' 

        obs[predictand_name] = utils.get_predictand(DATA_PATH, predictand_name, 'tasmean')
        obs[predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs[predictand_name] = utils.mask_data(
                    path = f'{DATA_PATH}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs[predictand_name],
                    secondGrid = obs[predictand_name])
        whole_obs['annual'][predictand_name] = obs[predictand_name]
        whole_obs_metrics['annual'][predictand_name] = utils.get_metrics_temp(whole_obs['annual'][predictand_name], short = True)
        for metric_stat in metrics:
            total_metrics[metric_stat].append(whole_obs_metrics['annual'][predictand_name][metric_stat])
    
    utils.create_multi_plot(
            data=whole_obs_metrics['annual'],
            vmin=[5, 15], vmax=[25, 35],
            fig_path=FIGS_PATH, fig_name=f'fig1_metrics_observation_annual_{yearsTrain[0]}-{yearsTest[1]}.pdf',
            n_rows=2, n_cols=len(whole_obs_metrics['annual']),
            color='hot_r',
            cmap_colors=[(0, 1, 11)]*2, cmap_first_color=[1]*2,
            cmap_min=[0]*2, cmap_max=[6]*2,
            x_map=predictands_map, y_map=metrics_map,
            var='tasmean', fontsize=18
    )
    del whole_obs, whole_obs_metrics, obs

    # Standard Deviation all predictands for Mean and P99
    total_metrics_concatened = {}
    std_metrics = {'all': {}}

    for metric_stat in metrics:    
        total_metrics_concatened[metric_stat] = xr.concat(total_metrics[metric_stat], dim='member')
        std_metrics['all'][metric_stat] = total_metrics_concatened[metric_stat].std(dim='member')

     # Plots standard deviation
    utils.create_multi_plot(
            data=std_metrics,
            vmin=[0, 0], vmax=[1.5, 1.5], color = 'magma_r',
            fig_path=FIGS_PATH, fig_name=f'fig1_metrics_standard_deviation_whole.pdf',
            n_rows=2, n_cols=len(std_metrics),
            cmap_colors=[(0, 1, 11)]*2, cmap_first_color=(1, 1),
            cmap_min=(0, 0), cmap_max=(6, 6), title='Std'
    )

            
    del total_metrics, total_metrics_concatened, std_metrics
    print("Figura 1 completada!")

    
    ### # FIG2 # ####
if '2' in FIGS:
    stat_metrics = ['rmse', 'bias', 'bias99']
    stats_map = {'rmse': 'RMSE', 'bias': 'Bias-mean', 'bias99': 'Bias-99th'}
    vminMetric = {'rmse': (0, 0, 20), 'bias': (-2.0, 0, 20), 'bias99': (-2.0, 0, 20)}
    vmaxMetric = {'rmse': (2.0, 20, 20), 'bias': (2, 20, 20), 'bias99': (2.0, 20, 20)}
    
    

    predictands_total_mean = []
    predictands_group_mean = []
    stat_realization_mean = {f'{metric}': {} for metric in stat_metrics}
    error_std = {f'{stat}': {'all': []} for stat in stat_metrics}
    for i, predictand_name in enumerate(predictands):      
        modelName = f'DeepESD_tas_{predictand_name}' 
        loaded_test_obs = utils.get_predictand(DATA_PATH, predictand_name, 'tasmean')
        loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
        loaded_test_obs = utils.mask_data(
                    path = f'{DATA_PATH}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = loaded_test_obs,
                    secondGrid = loaded_test_obs)
        
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        
        stat_realizations = {f'{stat}': [] for stat in stat_metrics}

        for predictand_number in predictand_numbered:
            modelName = f'deepesd_{predictand_number}_2004-01-01-2015-12-31'
            loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}.nc')
            rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time']))['tasmean'])
            bias = (loaded_test.mean(['time'])['tasmean'] - loaded_test_obs.mean(['time'])['tasmean'])
            loaded_test_99 = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
            loaded_test_obs_99 = loaded_test_obs.resample(time = 'YE').quantile(0.99, dim = 'time')
            bias_99 = (loaded_test_99.mean(['time'])['tasmean'] - loaded_test_obs_99.mean(['time'])['tasmean'])

            stat_realizations[stat_metrics[0]].append(rmse)
            stat_realizations[stat_metrics[1]].append(bias)
            stat_realizations[stat_metrics[2]].append(bias_99)

        for stat_name, realization_list in stat_realizations.items():
            stat_concat = xr.concat(realization_list, dim='member')
            stat_realization_mean[stat_name][predictand_name] = stat_concat.mean(dim='member')

    figName = f'fig2_rmse_bias_error_{ENSEMBLE_QUANTITY}.pdf'
    # Create fig and axes
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
    inverted_stat_realization = {k: {i: d[k] for i, d in stat_realization_mean.items()} for k in next(iter(stat_realization_mean.values()))}
    rows = len(stat_metrics)

    utils.create_multi_plot(
            data = inverted_stat_realization,
            vmin=[0, -2, -2], vmax=[2, 2, 2],
            fig_path=FIGS_PATH, fig_name=figName, 
            n_rows=rows, n_cols=len(predictands),
            cmap_colors=[(0, 1, 20)]*rows,
            cmap_first_color=[0]*rows, cmap_last_color=[20]*rows,
            color=['Reds', 'RdBu_r', 'RdBu_r'],
            cmap_min=(0,)*rows, cmap_max=(6,)*rows,
            orientation='horizontal', spacing='uniform',
            var=None, fontsize=18,
            title=None, x_map=predictands_map, y_map=stats_map
    )



    vminMetric = {'rmse': (0.0, 0, 9), 'bias': (0, 0, 9), 'bias99': (0.0, 0, 9)}
    vmaxMetric = {'rmse': (1.0, 9, 9), 'bias': (1.0, 9, 9), 'bias99': (1.0, 9, 9)}
 
    for j, stat_name in enumerate(stat_metrics):
        error_std = {stat_name: {'all': None}}
        rmse_bias_mean_list = {key: list(value.values()) for key, value in stat_realization_mean.items()}
        error_std[stat_name]['all'] = xr.concat(rmse_bias_mean_list[stat_name], dim='member').std(dim='member')
        title = 'Std' if j==0 else None
        utils.create_multi_plot(
                data = error_std,
                vmin=[0, 0, 0], vmax=[0.8, 0.8, 0.8],
                fig_path=FIGS_PATH, fig_name=f'fig2_std_bias_error_{ENSEMBLE_QUANTITY}_{stat_name}.pdf', 
                n_rows=1, n_cols=1,
                cmap_colors=[(0, 1, 9)],
                cmap_first_color=[1],
                color='magma_r',
                cmap_min=[0], cmap_max=[5],
                orientation='horizontal', spacing='uniform',
                var=None, fontsize=18, y_height = 0.065,
                title=title
        )
    
    print("Figura 2 completada!")




### # FIG3 # ####
if '3' in FIGS:

    metric_label = {'mean': 'Mean', 'std': 'Std', '99mean': '99th-Mean'}

    metrics = ['mean', '99mean']
    # climatology - CCSIGNAL
    vminMetric = {'mean': {'mean':4.0, 'std': 0.00, 'm-ticks':1, 'std-ticks':1, 'm-cmap':1, 'std-cmap':0, 'std-cmap-short':0}, 
                  '99mean': {'99mean':4.0, 'std': 0.00,  'm-ticks':1, 'std-ticks':1,'m-cmap':1, 'std-cmap':0, 'std-cmap-short':0}}
    vmaxMetric = {'mean': {'mean':11.0, 'std': 0.90, 'm-ticks':9, 'std-ticks':11, 'm-cmap':22, 'std-cmap':9, 'std-cmap-short':4}, 
                  '99mean': {'99mean':11.0, 'std': 0.90, 'm-ticks':9, 'std-ticks':11, 'm-cmap':22, 'std-cmap':9, 'std-cmap-short':5}}
    
    data_predictands = {key: {predictand_name: {} for predictand_name in predictands} for key in metrics}
    for metric in metrics:
        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part1.pdf'
        # Create fig and axes
        fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
       
        predictands_total_mean = []
        predictands_group_mean = []

        for i, predictand_name in enumerate(predictands):
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            # Reference Data
            hist_list = []

            for predictand_number in predictand_numbered:
                modelName = f'deepesd_{predictand_number}' 
                # Load hist inferences
                hist_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{gcm_ref_years[0]}-{gcm_ref_years[1]}.nc')
                if metric == '99mean':
                    hist_data = hist_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                hist_mean_time = hist_data.mean(dim='time')
                hist_list.append(hist_mean_time)

            # Future Data
            mean_list = []

            for predictand_number in predictand_numbered:
                modelName = f'deepesd_{predictand_number}' 
                # Load future inferences
                loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
                loaded_data = loaded_data.sel(time=slice(*future_3))
                if metric == '99mean':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                mean_time = loaded_data.mean(dim='time')
                mean_list.append(mean_time)
                
            predictand_data = {metric: None, 'std': None}

            predictand_data_ensemble = xr.concat(mean_list, dim='member') - xr.concat(hist_list, dim='member')

            predictand_data[metric] = predictand_data_ensemble.mean('member')
            predictand_data['std'] = predictand_data_ensemble.std('member')
            predictands_total_mean.append(predictand_data[metric])
            
            data_predictands[metric][predictand_name][metric] = predictand_data[metric]
            data_predictands[metric][predictand_name]['std'] = predictand_data['std']
            del predictand_data_ensemble, predictand_data


        utils.create_multi_plot(
                data = data_predictands[metric],
                vmin=[vminMetric[metric][metric], vminMetric[metric]['std']],
                vmax=[vmaxMetric[metric][metric], vmaxMetric[metric]['std']],
                fig_path=FIGS_PATH, fig_name=figName, 
                n_rows=len(metrics), n_cols=len(data_predictands[metric]),
                cmap_colors=((0, 1, 22), (0, 1, 9)),
                cmap_first_color=(vminMetric[metric]['m-cmap'], vminMetric[metric]['std-cmap']),
                cmap_last_color=(vmaxMetric[metric]['m-cmap'], vmaxMetric[metric]['std-cmap']),
                color=['hot_r', 'magma_r'],
                cmap_min=(vminMetric[metric]['m-ticks'], vminMetric[metric]['std-ticks']),
                cmap_max=(vmaxMetric[metric]['m-ticks'], vmaxMetric[metric]['std-ticks']), 
                orientation='horizontal', spacing='uniform',
                var='tasmean', fontsize=18,
                x_map=predictands_map, y_map=metric_label)

        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part2.pdf'
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        data_std = {'std': {'all-std': mean_combined.std(dim='member')}}
        utils.create_multi_plot(
                data = data_std,
                vmin=[vminMetric[metric]['std']], vmax=[vmaxMetric[metric]['std']],
                fig_path=FIGS_PATH, fig_name=figName, 
                n_rows=1, n_cols=1,
                cmap_colors=[(0, 1, 10)],
                cmap_first_color=[vminMetric[metric]['std-cmap']],
                cmap_last_color=[vmaxMetric[metric]['std-cmap']],
                color='magma_r',
                cmap_min=[vminMetric[metric]['std-cmap-short']],
                cmap_max=[vmaxMetric[metric]['std-cmap-short']], 
                orientation='horizontal', spacing='uniform',
                var='tasmean', fontsize=18, y_height = 0.035,
                title='Std'
        )
        

    del mean_combined, predictands_total_mean, predictands_group_mean, mean_list
    print("Figura 3 completada!")


### # FIG 4 # ####
if '4' in FIGS:
    shape_name_list = ['Iberia', 'Pyrenees', "Sierra Nevada", "Ebro Valley", "Guadalquivir Valley", "Central Plateau", "Cantabrian Mountains"]
    #*********************************************************************+
    references_grid = {shape: [] for shape in shape_name_list}
    reference_grid = xr.open_dataset(f'{PREDS_PATH}/predGCM_deepesd_AEMET_0.25deg_1_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
    reference_grid = reference_grid.sel(time=slice(yearsLong[0],'2081-01-02'))
    for shape in shape_name_list:
        if shape == 'Iberia':
            references_grid[shape] = None
        else:
            region = utils.get_region(shape, regions)
            references_grid[shape] = reference_grid.sel(lon=slice(region[3], region[4]), lat=slice(region[1], region[2]))



    # Boxplot for short, medium and long ccsignal
    periods = [future_2, future_4, future_3]
    xmin = (1.0, 1.0)
    xmax = (11.0, 11.5)
    # CC SIGNAL
    for shape in shape_name_list:
        # Labels
        colors = ['darkgreen', 'darkblue', 'darkred']
        names = ['Short', 'Medium', 'Long']
        legend_handles = []

        figName = f'fig4_boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
        # Create fig and axes
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax = ax1.twiny()
        ax_99 = ax1

        # Plots every set of data for the period in the same plot
        for i, period in enumerate(periods):
            data_to_plot = []
            data_to_plot_99 = []
            for predictand_name in predictands:
                   

                predictand_data = []
                ccsignal_predictand = []
                ccsignal_predictand_99 = []
                predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

                for predictand_number in predictand_numbered:
                    modelName = f'deepesd_{predictand_number}' 
                    # FUTURE DATA
                    loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsGCM[i][0]}-{yearsGCM[i][1]}.nc')
                    loaded_data = loaded_data.sel(time=slice(*(period)))
                    grided_data = loaded_data.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else loaded_data

                    grided_data_99 = grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                    grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
                    grided_mean_99 = grided_data_99.mean(dim=['time', 'lat', 'lon'])
                    # REFERENCE DATA
                    ref_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{gcm_ref_years[0]}-{gcm_ref_years[1]}.nc')

                    ref_grided_data = ref_data.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else ref_data

                    ref_grided_data_99 = ref_grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                    ref_grided_mean = ref_grided_data.mean(dim=['time', 'lat', 'lon']) 
                    ref_grided_mean_99 = ref_grided_data_99.mean(dim=['time', 'lat', 'lon'])
                    # CC SIGNAL
                    ccsignal_predictand.append(grided_mean- ref_grided_mean)
                    ccsignal_predictand_99.append(grided_mean_99 - ref_grided_mean_99)


                ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand])
                data_to_plot.append(ccsignal_array)
                ccsignal_array_99 = np.array([ds['tasmean'].values for ds in ccsignal_predictand_99])
                data_to_plot_99.append(ccsignal_array_99)

            color = colors[i]
            bplot = ax.boxplot(data_to_plot, positions= 4+ np.arange(len(predictands)), widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=2))
            ax.set_xlim(xmin[1], xmax[1])
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks(np.linspace(xmin[0], xmax[0], 11))
            ax.tick_params(axis='x', labelsize=16)
 

            bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands))+0.1 , widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=1.5))
            ax_99.set_xlim(xmin[1], xmax[1])
            ax_99.xaxis.set_ticks_position('bottom')
            ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 11))
            ax_99.tick_params(axis='x', labelsize=16)

            
                # Assign label to X axis
            if i == 0:
                ax.set_xlabel(f'CC Signal Tasmean {shape}')
            legend_handles.append(bplot["boxes"][0])


        # Labels for Y axis
        ax1.set_yticks(np.arange(len(predictands)*2) )
        ax1.set_yticklabels(predictands_tick_names*2, fontsize=16)

        # Compute plot center
        y_min, y_max = ax1.get_ylim()
        y_center = (y_min + y_max) / 2
        # Draw horizontal line
        ax1.hlines(y=y_center, xmin=xmin[0], xmax=xmax[1], colors='black', linestyles='dashed', linewidth=1)

        # Add dotted grid
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax_99.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Add labels at upper-right corner
        ax1.text(xmax[1] - 0.7, y_max/2 + 0.5, 'Mean', fontsize=14, fontweight='bold', ha='right', va='top', color='black')
        ax_99.text(xmax[1] - 0.2, + 0.5, '99th Percentile', fontsize=14, fontweight='bold', ha='right', va='top', color='black')


        plt.legend(legend_handles, names, loc='upper right', prop={'size': 14}, frameon=False)
        plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')



    print("Figura 4 completada!")
    
if '5' in FIGS:
    figName = f'fig5_extremes_ccsignals_Ensemble{ENSEMBLE_QUANTITY}.pdf'
    graph_data = {}

    for i, predictand_name in enumerate(predictands):

        # Future Data
        predictand_data = {}
        predictand_data_mean = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99': None}
        number_min_max = {}

        for p_num in range(1, ENSEMBLE_QUANTITY+1):
            
            modelName = f'deepesd_{predictand_name}_{p_num}'

            loaded_data_ref = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{gcm_ref_years[0]}-{gcm_ref_years[1]}.nc')
            mean_time_ref = loaded_data_ref.mean(dim='time')
            gridded_mean_ref = loaded_data_ref.mean(dim=['time', 'lat', 'lon']) 
            loaded_data_ref_99 = loaded_data_ref.resample(time = 'YE').quantile(0.99, dim = 'time')
            mean_time_ref_99 = loaded_data_ref_99.mean(dim='time')
            gridded_mean_ref_99 = loaded_data_ref_99.mean(dim=['time', 'lat', 'lon']) 

            loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
            loaded_data = loaded_data.sel(time=slice(*(future_3[0], future_3[1])))

            grided_mean = loaded_data.mean(dim=['time', 'lat', 'lon']) - gridded_mean_ref
            mean_time = loaded_data.mean(dim='time') - mean_time_ref

            loaded_data_99 = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
            grided_mean_99 = loaded_data_99.mean(dim=['time', 'lat', 'lon']) - gridded_mean_ref_99
            mean_time_99 = loaded_data_99.mean(dim='time') - mean_time_ref_99   



            # CHECK MIN AND MAX AND SAVE MIN, MAX, MEAN FOR CCSIGNAL
            for key, value in {'min_mean': (mean_time, grided_mean['tasmean'].values),
                            'max_mean': (mean_time, grided_mean['tasmean'].values),
                            'min_99': (mean_time_99, grided_mean_99['tasmean'].values),
                            'max_99': (mean_time_99, grided_mean_99['tasmean'].values)}.items():
                if (predictand_data_mean[key] == None 
                    or (value[1] < predictand_data_mean[key] and 'min' in key)
                    or (value[1] > predictand_data_mean[key] and 'max' in key)):

                    predictand_data_mean[key] = value[1]
                    predictand_data[key] = value[0]
                    number_min_max[key] = p_num


        graph_data[predictand_name] = {
            'Rmax-Rmin (Mean)': predictand_data['max_mean'] - predictand_data['min_mean'],
            'Rmax-Rmin (99th)': predictand_data['max_99'] - predictand_data['min_99'],
        }

    # Plots
    utils.create_multi_plot(
            data = graph_data,
            rtick_bool=[True, True], ltick_bool=[True, True],
            vmin=[0, 0], vmax=[2.0, 4.0],
            fig_path=FIGS_PATH, fig_name=figName, 
            n_rows=2, n_cols=len(graph_data),
            cmap_colors=[(0, 1, 11)]*2,
            cmap_first_color=[1, 1], cmap_last_color=[6, 12],
            color='hot_r',
            cmap_min=[0]*2, cmap_max=[6]*2, 
            orientation='horizontal', spacing='uniform',
            var='tasmean', fontsize=15,
            x_map=predictands_map, y_map={'Rmax-Rmin (Mean)': 'Rmax-Rmin (Mean)', 'Rmax-Rmin (99th)': 'Rmax-Rmin (99th)'}
    )
 
    print("Figura 5 completada!")

### # FIG6 # ####
if '6' in FIGS:
    valuesMinMax = {'Mean': (13, 15, 19, 21), '99Percentile': (25, 28, 33, 36)}
    color_list = ['crimson', 'forestgreen', 'royalblue', 'orchid', 'cadetblue']

    for metric in ['Mean', '99Percentile']:
        observational_mean = {}
        test_pred_total = {}
        gcm_pred_total = {}

        for predictand_name in predictands:
            rmse_test = []
            test_pred = []
            gcm_pred = []

            loaded_test_obs = utils.get_predictand(DATA_PATH, predictand_name, 'tasmean')
            loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
            loaded_test_obs = utils.mask_data(
                        path = f'{DATA_PATH}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = loaded_test_obs,
                        secondGrid = loaded_test_obs)
            
            if metric == '99Percentile':
                observed_mean = loaded_test_obs.resample(time = 'YE').quantile(0.99, dim = 'time')
            else:
                observed_mean = loaded_test_obs

            observed_mean = observed_mean.mean(dim=['time', 'lat', 'lon'])['tasmean']
            observational_mean[predictands_map[predictand_name]] = observed_mean


            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            for predictand_number in predictand_numbered:
                modelName = f'deepesd_{predictand_number}'
                loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}_{yearsTest[0]}-{yearsTest[1]}.nc')
                rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time', 'lat', 'lon']))['tasmean'])
                rmse_test.append(rmse)

                if metric == '99Percentile':
                    loaded_test = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
                test_pred.append(loaded_test.mean(dim=['time', 'lat', 'lon'])['tasmean'])

                
                loaded_pred = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
                loaded_pred = loaded_pred.sel(time=slice(*(future_3[0], future_3[1])))
                if metric == '99Percentile':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
                gcm_pred.append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])

            # Plotting
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # Calculation of linear significance lines
            coefficients_test = np.polyfit(rmse_test, test_pred, 1)
            coefficients_gcm = np.polyfit(rmse_test, gcm_pred, 1)
            m_test, b_test = coefficients_test
            m_gcm, b_gcm = coefficients_gcm

            test_pred_total[predictands_map[predictand_name]] = test_pred
            gcm_pred_total[predictands_map[predictand_name]] = gcm_pred

        figName = f'fig6_temps_Mean_Ensemble{ENSEMBLE_QUANTITY}_{metric}'
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter for dataset 1      
        for color_num, predictand_name in enumerate(predictands):
            ax.scatter(test_pred_total[predictands_map[predictand_name]], gcm_pred_total[predictands_map[predictand_name]], color=f'{color_list[color_num]}', label=f'{predictand_name}', alpha=0.4)
            ax.plot([valuesMinMax[metric][0], valuesMinMax[metric][3]], [valuesMinMax[metric][0], valuesMinMax[metric][3]], color='black', linestyle='-', linewidth=1.5)

        # Labels and legend
        ax.set_xlim(valuesMinMax[metric][0], valuesMinMax[metric][3])
        ax.set_ylim(valuesMinMax[metric][0], valuesMinMax[metric][3])
        ax.set_xlabel('Temperature Test', fontsize=12)
        ax.set_ylabel('Temperature Long', fontsize=12)
        ax.set_title('Long vs Test', fontsize=14)
        ax.legend()

        # Show grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


        figName = f'fig6_zoom_Ensemble{ENSEMBLE_QUANTITY}_{metric}'
        # Plotting
        fig, ax = plt.subplots(figsize=(12, 9))

        # Scatter for dataset 1       
        for color_num, predictand_name in enumerate(predictands):
            test_values = test_pred_total[predictands_map[predictand_name]]
            gcm_values = gcm_pred_total[predictands_map[predictand_name]]
            ax.scatter(test_values, gcm_values, color=f'{color_list[color_num]}', label=f'{predictands_map[predictand_name]}', alpha=0.7)
            ax.plot([valuesMinMax[metric][0], valuesMinMax[metric][2]], [valuesMinMax[metric][1], valuesMinMax[metric][3]], color='black', linestyle='-', linewidth=1.5)
            ax.axvline(observational_mean[predictands_map[predictand_name]], color=f'{color_list[color_num]}', linestyle='--', linewidth=1)
            test_mean = np.mean([da.data for da in test_values])
            gcm_mean = np.mean([da.data for da in gcm_values])
            ax.scatter(test_mean, gcm_mean, color=f'{color_list[color_num]}', marker='+', s=100)

        # Labels and legend
        ax.set_xlim(valuesMinMax[metric][0], valuesMinMax[metric][1])
        ax.set_ylim(valuesMinMax[metric][2],valuesMinMax[metric][3])
        ax.set_xlabel('Temperature Test', fontsize=12)
        ax.set_ylabel('Temperature Long', fontsize=12)
        ax.set_title('Long vs Test', fontsize=14)
        ax.legend()


        # Show grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


    print("Figura 6 completada!")


if FIGS == '8':

    oro_path = f"{FIGS_PATH}geo_1279l4_0.1x0.1.grib2_v4_unpack.nc" #Era5 orography data
    ds = xr.open_dataset(oro_path)
    
    oro = ds["z"].squeeze()
    oro = oro.sel(latitude=slice(45, 35))
    oro = xr.concat([oro.sel(longitude=slice(350, 360)), oro.sel(longitude=slice(0, 5))], dim='longitude')

    if oro.max() > 10000:
        oro = oro / 9.81
        oro.attrs['units'] = 'm'


    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-10, 5, 35, 45], crs=ccrs.PlateCarree())
    

    im = ax.pcolormesh(
        oro['longitude'], oro['latitude'], oro,
        cmap='terrain', shading='auto', transform=ccrs.PlateCarree()
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.1)

    for name, lat_min, lat_max, lon_min, lon_max, color in regions:
        width = lon_max - lon_min
        height = lat_max - lat_min
        rect = Rectangle(
            (lon_min, lat_min), width, height,
            linewidth=1.5, edgecolor='black',
            facecolor=color, alpha=0.4,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(rect)

        position_ha = 'left' if name == 'Sierra Nevada' else 'center'
        position_va = 'top' if name == 'Sierra Nevada' else 'center'

        ax.text(
            lon_min + width / 2, lat_min + height / 2, name,
            fontsize=9, ha=position_ha, va=position_va,
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor='white', alpha=0.6)
        )

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
    cb.set_label("Orography (m)")

    plt.tight_layout()
    plt.savefig(f"{FIGS_PATH}/Toponomias_orography.pdf", bbox_inches='tight')
    plt.close()
    print("Figura 8 completada!")


if FIGS == 'extra':

    metrics = ['mean', '99mean']
    metric_label = {'mean': 'Mean', '99mean': '99th-Mean'}
    shorten_predictands = ['ERA5-Land0.25deg', 'E-OBS', 'CHELSA']

    data_predictands = {predictand_name:{metric: None for metric in metrics} for predictand_name in predictands}
    difference_data = {predictand_name:{metric: None for metric in metrics} for predictand_name in shorten_predictands}
    relative_data = {predictand_name:{metric: None for metric in metrics} for predictand_name in shorten_predictands}


    

    for metric in metrics:
        for predictand_name in predictands:
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)] 
            # Reference Data
            hist_list = []

            for predictand_number in predictand_numbered:
                modelName = f'deepesd_{predictand_number}' 
                # Load hist inferences
                hist_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{gcm_ref_years[0]}-{gcm_ref_years[1]}.nc')
                if metric == '99mean':
                    hist_data = hist_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                hist_mean_time = hist_data.mean(dim='time')
                hist_list.append(hist_mean_time)

            # Future Data
            mean_list = []

            for predictand_number in predictand_numbered:
                modelName = f'deepesd_{predictand_number}' 
                # Load future inferences
                loaded_data = xr.open_dataset(f'{PREDS_PATH}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
                loaded_data = loaded_data.sel(time=slice(*future_3))
                if metric == '99mean':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                mean_time = loaded_data.mean(dim='time')
                mean_list.append(mean_time)
                

            predictand_data_ensemble = xr.concat(mean_list, dim='member') - xr.concat(hist_list, dim='member')

            predictand_data = predictand_data_ensemble.mean('member')

            
            data_predictands[predictand_name][metric] = predictand_data
            print(f"{metric}-{predictand_name} done")
            del predictand_data_ensemble, predictand_data
        for predictand_name in shorten_predictands:
            difference_data[predictand_name][metric] = data_predictands[predictand_name][metric] - data_predictands['AEMET_0.25deg'][metric]
            relative_data[predictand_name][metric] = (difference_data[predictand_name][metric] / data_predictands['AEMET_0.25deg'][metric]) * 100

    figName = f'figExtra_difference_CCSignal_{ENSEMBLE_QUANTITY}_ROCIO-IBEB.pdf'
    utils.create_multi_plot(
        data = difference_data,
        vmin=[-2.5, -2.5],
        vmax=[2.5, 2.5],
        fig_path=FIGS_PATH, fig_name=figName, 
        n_rows=len(metrics), n_cols=len(difference_data),
        cmap_colors=((0, 1, 21), (0, 1, 21)),
        cmap_first_color=(1, 1),
        cmap_last_color=(21, 21),
        color=['RdYlBu_r', 'RdYlBu_r'],
        cmap_min=(1, 1),
        cmap_max=(12, 12), 
        orientation='horizontal', spacing='uniform',
        var='tasmean', fontsize=16,
        x_map=predictands_map, y_map=metric_label)

    figName = f'figExtra_relative_CCSignal_{ENSEMBLE_QUANTITY}_ROCIO-IBEB.pdf'
    utils.create_multi_plot(
        data = relative_data,
        vmin=[-25, -25],
        vmax=[25, 25],
        fig_path=FIGS_PATH, fig_name=figName, 
        n_rows=len(metrics), n_cols=len(relative_data),
        cmap_colors=((0, 1, 21), (0, 1, 21)),
        cmap_first_color=(1, 1),
        cmap_last_color=(21, 21),
        color=['BrBG', 'BrBG'],
        cmap_min=(1, 1),
        cmap_max=(12, 12), 
        orientation='horizontal', spacing='uniform',
        var='tasmean', fontsize=16,
        x_map=predictands_map, y_map=metric_label)
    
    print("Figura extra completada!")

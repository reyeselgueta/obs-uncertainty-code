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
PREDS_PATH_GCM = './preds/gcm/'
PREDS_PATH_TEST = './preds/test/'
REF_GRID = 'AEMET_0.25deg_tasmean_1951-2022.nc' # Put a dataset of your choice that is a common grid for your predictands.

# # INPUT DATA
FIGS = sys.argv[1]
ENSEMBLE_QUANTITY = 10
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'


# # GENERAL VARIABLES
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']
predictands_map = {'ERA5-Land0.25deg': 'ERA5-Land', 'E-OBS': 'E-OBS','AEMET_0.25deg':'ROCIO-IBEB', 'Iberia01_v1.0':'Iberia01', 'CHELSA': 'CHELSA'}

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



### # FIG1 # ####
if '1' in FIGS:
# DATOS OBSERVACION
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
        obs[predictand_name] = utils.mask_data( # TODO Reemplazar a uso de d4d
                    path = f'{DATA_PATH}{REF_GRID}',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs[predictand_name],
                    secondGrid = obs[predictand_name])
        whole_obs['annual'][predictand_name] = obs[predictand_name]
        whole_obs_metrics['annual'][predictand_name] = utils.get_metrics_temp(whole_obs['annual'][predictand_name], short = True)
        for metric_stat in metrics:
            total_metrics[metric_stat].append(whole_obs_metrics['annual'][predictand_name][metric_stat])
    
    utils.create_multi_graph(
            data=whole_obs_metrics['annual'],
            vmin=[5, 15], vmax=[25, 35],
            fig_path=FIGS_PATH, fig_name=f'fig1_metrics_observation_annual_{yearsTrain[0]}-{yearsTest[1]}.pdf',
            n_rows=2, n_cols=len(whole_obs_metrics['annual']),
            color='hot_r',
            cmap_colors=[(0, 1, 11)]*2, cmap_first_color=[1]*2,
            cmap_min=[0]*2, cmap_max=[6]*2, tick_bool=False, 
            x_map=predictands_map, y_map=metrics_map,
            var='tasmean', fontsize=16
    )
    del whole_obs, whole_obs_metrics, obs

    # Standard Deviation all predictands for Mean and P99
    total_metrics_concatened = {}
    std_metrics = {'all': {}}

    for metric_stat in metrics:    
        total_metrics_concatened[metric_stat] = xr.concat(total_metrics[metric_stat], dim='member')
        std_metrics['all'][metric_stat] = total_metrics_concatened[metric_stat].std(dim='member')

     # GRAPHS STANDAR DEVIATION
    utils.create_multi_graph(
            data=std_metrics,
            vmin=[0, 0], vmax=[1.5, 1.5],
            fig_path=FIGS_PATH, fig_name=f'fig1_metrics_standard_deviation_whole.pdf',
            n_rows=2, n_cols=len(std_metrics),
            cmap_colors=[(0, 1, 11)]*2, cmap_first_color=(1, 1),
            cmap_min=(0, 0), cmap_max=(6, 6), tick_bool=False, title='Std'
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
                    path = f'{DATA_PATH}{REF_GRID}',
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
    # Crear la figura y los ejes
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
    inverted_stat_realization = {k: {i: d[k] for i, d in stat_realization_mean.items()} for k in next(iter(stat_realization_mean.values()))}
    rows = len(stat_metrics)

    utils.create_multi_graph(
            data = inverted_stat_realization,
            vmin=[0, 0, 0], vmax=[2, 2, 2],
            fig_path=FIGS_PATH, fig_name=figName, 
            n_rows=rows, n_cols=len(predictands),
            cmap_colors=[(0, 1, 20)]*rows,
            cmap_first_color=[0]*rows, cmap_last_color=[20]*rows,
            color=['Reds', 'RdBu_r', 'RdBu_r'],
            cmap_min=(0,)*rows, cmap_max=(6,)*rows, tick_bool=False, 
            orientation='horizontal', spacing='uniform',
            var=None, fontsize=15,
            title=None, x_map=predictands_map, y_map=stats_map
    )



    vminMetric = {'rmse': (0.0, 0, 10), 'bias': (0, 0, 10), 'bias99': (0.0, 0, 10)}
    vmaxMetric = {'rmse': (1.0, 10, 10), 'bias': (1.0, 10, 10), 'bias99': (1.0, 10, 10)}
 
    for j, stat_name in enumerate(stat_metrics):
        error_std = {stat_name: {'all': None}}
        rmse_bias_mean_list = {key: list(value.values()) for key, value in stat_realization_mean.items()}
        error_std[stat_name]['all'] = xr.concat(rmse_bias_mean_list[stat_name], dim='member').std(dim='member')
        title = 'Std' if j==0 else None
        utils.create_multi_graph(
                data = error_std,
                vmin=[0, 0, 0], vmax=[1, 1, 1],
                fig_path=FIGS_PATH, fig_name=f'fig2_std_bias_error_{ENSEMBLE_QUANTITY}_{stat_name}.pdf', 
                n_rows=1, n_cols=1,
                cmap_colors=[(0, 1, 11)],
                cmap_first_color=[1],
                color='cool',
                cmap_min=[0], cmap_max=[6], tick_bool=False, 
                orientation='horizontal', spacing='uniform',
                var=None, fontsize=15,
                title=title
        )
    
    print("Figura 2 completada!")
   


### # FIG3 # ####
if '3' in FIGS:
    

    metric_label = {'mean': 'Mean', 'std': 'Std'}

    metrics = ['Mean', '99Percentile']
    # climatology - CCSIGNAL
    vminMetric = {'Mean': {'mean':4.6, 'std': 0.05, 'm-ticks':4, 'std-ticks':0, 'm-cmap':4, 'std-cmap':0, 'std-cmap-short':0}, 
                  '99Percentile': {'mean':5, 'std': 0.25,  'm-ticks':2, 'std-ticks':2,'m-cmap':2, 'std-cmap':2, 'std-cmap-short':0}}
    vmaxMetric = {'Mean': {'mean':8.6, 'std': 0.85, 'm-ticks':15, 'std-ticks':9, 'm-cmap':14, 'std-cmap':8, 'std-cmap-short':5}, 
                  '99Percentile': {'mean':13, 'std': 1.25, 'm-ticks':23, 'std-ticks':13, 'm-cmap':22, 'std-cmap':13, 'std-cmap-short':5}}
    
    data_predictands = {key: {predictand_name: {} for predictand_name in predictands} for key in metrics}
    for metric in metrics:
        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part1.pdf'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
       
        predictands_total_mean = []
        predictands_group_mean = []

        for i, predictand_name in enumerate(predictands):
            # Historical Data
            obs_predictand = utils.get_predictand(f'{DATA_PATH}', predictand_name, 'tasmean')
            obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
            obs_predictand = utils.mask_data(
                        path = f'{DATA_PATH}{REF_GRID}',
                        var='tasmean',
                        to_slice=(yearsTrain[0], yearsTest[1]),
                        objective = obs_predictand.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                        secondGrid = obs_temp)
            if metric == '99Percentile':
                obs_predictand = obs_predictand.resample(time = 'YE').quantile(0.99, dim = 'time')
            obs_predictand_mean = obs_predictand.mean(dim='time')

            # Future Data
            predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
            predictand_data = {'mean': None, 'std': None}
            mean_list = []

            grided_mean_list = []
            for predictand_number in predictand_numbered:
                modelName = f'deepesd_{predictand_number}' 
                loaded_data = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
                loaded_data = loaded_data.sel(time=slice(*future_3))
                if metric == '99Percentile':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                mean_time = loaded_data.mean(dim='time')
                mean_list.append(mean_time)

            predictand_data_ensemble = xr.concat(mean_list, dim='member') - obs_predictand_mean

            predictand_data['mean'] = predictand_data_ensemble.mean('member')
            predictand_data['std'] = predictand_data_ensemble.std('member')
            predictands_total_mean.append(predictand_data['mean'])
            
            data_predictands[metric][predictand_name]['mean'] = predictand_data['mean']
            data_predictands[metric][predictand_name]['std'] = predictand_data['std']
            del predictand_data_ensemble, predictand_data


        utils.create_multi_graph(
                data = data_predictands[metric],
                vmin=[vminMetric[metric]['mean'], vminMetric[metric]['std']],
                vmax=[vmaxMetric[metric]['mean'], vmaxMetric[metric]['std']],
                fig_path=FIGS_PATH, fig_name=figName, 
                n_rows=len(metrics), n_cols=len(data_predictands[metric]),
                cmap_colors=((0, 1, 23), (0, 1, 12)),
                cmap_first_color=(vminMetric[metric]['m-cmap'], vminMetric[metric]['std-cmap']),
                cmap_last_color=(vmaxMetric[metric]['m-cmap'], vmaxMetric[metric]['std-cmap']),
                color=['hot_r', 'cool'],
                cmap_min=(vminMetric[metric]['m-ticks'], vminMetric[metric]['std-ticks']),
                cmap_max=(vmaxMetric[metric]['m-ticks'], vmaxMetric[metric]['std-ticks']), 
                tick_bool=False, 
                orientation='horizontal', spacing='uniform',
                var='tasmean', fontsize=16,
                x_map=predictands_map, y_map=metric_label)

        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part2.pdf'
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        data_std = {'std': {'all-std': mean_combined.std(dim='member')}}
        utils.create_multi_graph(
                data = data_std,
                vmin=[vminMetric[metric]['std']], vmax=[vmaxMetric[metric]['std']],
                fig_path=FIGS_PATH, fig_name=figName, 
                n_rows=1, n_cols=1,
                cmap_colors=[(0, 1, 11)],
                cmap_first_color=[vminMetric[metric]['std-cmap']],
                cmap_last_color=[vmaxMetric[metric]['std-cmap']],
                color='cool',
                cmap_min=[vminMetric[metric]['std-cmap-short']],
                cmap_max=[vmaxMetric[metric]['std-cmap-short']], 
                tick_bool=False, 
                orientation='horizontal', spacing='uniform',
                var='tasmean', fontsize=16,
                title='Std'
        )
        

    del mean_combined, predictands_total_mean, predictands_group_mean, mean_list
    print("Figura 3 completada!")


### # FIG 4 # ####
if '4' in FIGS:
    shape_name_list = ['Iberia', 'Pirineos']
    #*********************************************************************+
    references_grid = {shape: [] for shape in shape_name_list}
    reference_grid = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_deepesd_AEMET_0.25deg_1_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
    reference_grid = reference_grid.sel(time=slice(yearsLong[0],'2081-01-02'))
    for shape in shape_name_list:
        if shape == 'Iberia':
            references_grid[shape] = None
        elif shape == 'Pirineos':
            references_grid[shape] = reference_grid.sel(lon=slice(-0.37, 3.37), lat=slice(41.42, 42.80))
        elif shape == 'Duero':
            references_grid[shape] = reference_grid.sel(lon=slice(-6.59, -4.75), lat=slice(40.85, 42.45))
        elif shape == 'Tinto':
            references_grid[shape] = reference_grid.sel(lon=slice(-7.23, -5.20), lat=slice(36.00, 38.20))


    # GRAFICOS BOXPLOT PARA SHORT, MEDIUM y LONG / CCSIGNAL
    periods = [future_2, future_4, future_3]
    xmin = (1.0, 1.0)
    xmax = (11.0, 11.5)
    # CC SIGNAL
    for shape in shape_name_list:
        # Etiquetas
        colors = ['darkgreen', 'darkblue', 'darkred']
        names = ['Short', 'Medium', 'Long']
        legend_handles = []

        figName = f'fig4_boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
        # Crear la figura y los ejes
        fig, ax1 = plt.subplots(figsize=(20, 12))
        ax = ax1.twiny()
        ax_99 = ax1

        # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
        for i, period in enumerate(periods):
            data_to_plot = []
            data_to_plot_99 = []
            for predictand_name in predictands:
                obs2 = utils.get_predictand(f'{DATA_PATH}', predictand_name, 'tasmean')
                obs_temp = obs2.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
                obs2 = utils.mask_data(
                            path = f'{DATA_PATH}{REF_GRID}',
                            var='tasmean',
                            to_slice=(hist_baseline[0], hist_baseline[1]),
                            objective = obs2.sel(time=slice(*(hist_baseline))),
                            secondGrid = obs_temp)
                obs2 = obs2.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else obs2
                obs2_99 = obs2.resample(time = 'YE').quantile(0.99, dim = 'time')

                obs2_mean = obs2.mean(dim=['time', 'lat', 'lon'])
                obs2_99_mean = obs2_99.mean(dim=['time', 'lat', 'lon'])


                predictand_data = []
                ccsignal_predictand = []
                ccsignal_predictand_99 = []
                predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]

                for predictand_number in predictand_numbered:
                    modelName = f'deepesd_{predictand_number}' 
                    loaded_data = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsGCM[i][0]}-{yearsGCM[i][1]}.nc')
                    loaded_data = loaded_data.sel(time=slice(*(period)))
                    grided_data = loaded_data.sel(
                        lat=references_grid[shape].lat,
                        lon=references_grid[shape].lon,
                    ) if shape != 'Iberia' else loaded_data
                    
                    grided_data_99 = grided_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                    grided_mean = grided_data.mean(dim=['time', 'lat', 'lon']) 
                    grided_mean_99 = grided_data_99.mean(dim=['time', 'lat', 'lon'])
                    ccsignal_predictand.append(grided_mean- obs2_mean)
                    ccsignal_predictand_99.append(grided_mean_99 - obs2_99_mean)


                ccsignal_array = np.array([ds['tasmean'].values for ds in ccsignal_predictand])
                data_to_plot.append(ccsignal_array)
                ccsignal_array_99 = np.array([ds['tasmean'].values for ds in ccsignal_predictand_99])
                data_to_plot_99.append(ccsignal_array_99)

            color = colors[i]
            bplot = ax.boxplot(data_to_plot, positions= 5 + np.arange(len(predictands)), widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=2))
            ax.set_xlim(xmin[1], xmax[1]-3)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks(np.linspace(xmin[0], xmax[0]-3, 8))
            ax.tick_params(axis='x', labelsize=16)
 

            bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands))+0.1 , widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=1.5))
            ax_99.set_xlim(xmin[1], xmax[1])
            ax_99.xaxis.set_ticks_position('bottom')
            ax_99.set_xticks([]) if i>0 else ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 11))
            ax_99.tick_params(axis='x', labelsize=16)

            
            
            # Asignar la etiqueta del eje X solo para el primer eje (ax1)
            if i == 0:
                ax.set_xlabel(f'CC Signal Tasmean {shape}')
            # Crear un handle de la leyenda solo en la primera iteración para cada conjunto de datos
            legend_handles.append(bplot["boxes"][0])


        # Etiquetas del eje Y solo en ax1
        ax1.set_yticks(np.arange(len(predictands)*2) )
        ax1.set_yticklabels(list(predictands_map.values())*2, fontsize=16)

        # Calcular el centro del gráfico
        y_min, y_max = ax1.get_ylim()
        y_center = (y_min + y_max) / 2
        # Dibujar una línea horizontal
        ax1.hlines(y=y_center, xmin=xmin[1], xmax=xmax[1], colors='black', linestyles='dashed', linewidth=1)

        # Agregar la cuadrícula punteada
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax_99.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Agregar etiquetas en la parte superior derecha
        ax1.text(xmax[1] - 1, y_max/2 + 0.5, 'Mean', fontsize=14, fontweight='bold', ha='right', va='top', color='black')
        ax_99.text(xmax[1] - 0.5, + 0.5, '99th Percentile', fontsize=14, fontweight='bold', ha='right', va='top', color='black')


        plt.legend(legend_handles, names, loc='upper right', prop={'size': 14}, frameon=False)
        plt.savefig(f'{FIGS_PATH}/{figName}.pdf', bbox_inches='tight')


    print("Figura 4 completada!")
    
### # FIG5 # ####
if '5' in FIGS:
    figName = f'fig5_extremes_ccsignals_Ensemble{ENSEMBLE_QUANTITY}.pdf'
    graph_data = {}

    for i, predictand_name in enumerate(predictands):
        # Observed Data
        obs_predictand = utils.get_predictand(f'{DATA_PATH}', predictand_name, 'tasmean')
        obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_predictand = utils.mask_data(
                    path = f'{DATA_PATH}{REF_GRID}',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                    secondGrid = obs_temp)

        obs_predictand_mean = obs_predictand.mean(dim='time')
        obs_predictand_mean_99 = obs_predictand.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')


        # Future Data
        predictand_data = {}
        predictand_data_mean = {'min_mean': None, 'max_mean': None, 'min_99': None, 'max_99': None}
        number_min_max = {}

        for p_num in range(1, ENSEMBLE_QUANTITY+1):
            
            modelName = f'deepesd_{predictand_name}_{p_num}'
            loaded_data = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
            loaded_data = loaded_data.sel(time=slice(*(future_3[0], future_3[1])))
            

            grided_mean = loaded_data.mean(dim=['time', 'lat', 'lon']) 
            mean_time = loaded_data.mean(dim='time')

            loaded_data_99 = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
            grided_mean_99 = loaded_data_99.mean(dim=['time', 'lat', 'lon']) 
            mean_time_99 = loaded_data_99.mean(dim='time')


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

    # GRAFICOS
    utils.create_multi_graph(
            data = graph_data,
            vmin=[0, 0], vmax=[2, 4],
            fig_path=FIGS_PATH, fig_name=figName, 
            n_rows=2, n_cols=len(graph_data),
            cmap_colors=[(0, 1, 11)]*2,
            cmap_first_color=[1, 1], cmap_last_color=[6, 11],
            color='hot_r',
            cmap_min=[0]*2, cmap_max=[6]*2, tick_bool=False, 
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
                        path = f'{DATA_PATH}{REF_GRID}',
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

                
                loaded_pred = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{yearsLong[0]}-{yearsLong[1]}.nc')
                loaded_pred = loaded_pred.sel(time=slice(*(future_3[0], future_3[1])))
                if metric == '99Percentile':
                    loaded_pred = loaded_pred.resample(time = 'YE').quantile(0.99, dim = 'time')
                gcm_pred.append(loaded_pred.mean(dim=['time', 'lat', 'lon'])['tasmean'])

            # Plotting
            fig, ax1 = plt.subplots(figsize=(8, 6))

            # Cálculo de las líneas de significancia
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


if FIGS=='7':
    # Crear figura y ejes
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-10, 5, 35, 45])

    # Elementos geográficos
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Definir regiones como rectángulos: (nombre, lat_min, lat_max, lon_min, lon_max)
    regions = [
        ("Central Plateau", 39.0, 41.5, -6.5, -2.5, "lightgrey"),
        ("Cantabrian Mountains", 42.5, 43.5, -7.0, -2.5, "lightgrey"),
        ("Ebro Valley", 40.5, 42.5, -2.0, 1.5, "lightgrey"),
        ("Pyrenees", 41.42, 42.8, -0.37, 3.37, "lightgrey"),
        ("Sierra Nevada", 36.7, 37.2, -3.6, -2.8, "lightgrey"),
        ("Guadalquivir Valley", 36.8, 38.5, -6.5, -3.5, "lightgrey")
    ]

    # Añadir regiones como rectángulos de color
    for name, lat_min, lat_max, lon_min, lon_max, color in regions:
        width = lon_max - lon_min
        height = lat_max - lat_min
        rect = Rectangle((lon_min, lat_min), width, height,
                        linewidth=1.5, edgecolor='black',
                        facecolor=color, alpha=0.4,
                        transform=ccrs.PlateCarree())
        ax.add_patch(rect)
        # Etiqueta en el centro de cada región
        position_ha = 'left' if name=='Sierra Nevada' else 'center'
        position_va = 'top' if name=='Sierra Nevada' else 'center'
            
        ax.text(lon_min + width / 2, lat_min + height / 2, name,
                fontsize=9, ha=position_ha, va=position_va,
                transform=ccrs.PlateCarree(), bbox=dict(facecolor='white', alpha=0.6))

    # Cuadrícula
    ax.gridlines(draw_labels=True)

    # Título y guardar
    plt.tight_layout()
    plt.savefig(f"{FIGS_PATH}Toponomias.pdf", bbox_inches='tight')
    plt.close()

    print("Figura 7 completada!")



import xarray as xr
import utils
import numpy as np
import matplotlib.pyplot as plt



# DATA_PATH_PREDICTORS = '/lustre/gmeteo/PTICLIMA/DATA/PROJECTIONS/CMIP6_PNACC/CMIP6_models/'
# DATA_PATH_PREDICTANDS_READ = '/lustre/gmeteo/PTICLIMA/DATA/AUX/GRID_INTERCOMP/'
DATA_PATH_PREDICTANDS_SAVE = '/lustre/gmeteo/WORK/reyess/data/predictand/'# TODO MERGIAR
# DATA_PATH_SHAPE = '/lustre/gmeteo/WORK/reyess/shapes/'
FIGS_PATH = '/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/figures/'
MODELS_PATH = '/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/models/'
# DATA_PREDICTORS_TRANSFORMED = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
PREDS_PATH_GCM = '/oceano/gmeteo/users/reyess/paper1-code/preds/gcm/'# TODO MERGIAR
PREDS_PATH_TEST = '/oceano/gmeteo/users/reyess/paper1-code/preds/test/' #TODO MERGIAR

# # INPUT DATA
FIGS = '2'#str(sys.argv[1])
ENSEMBLE_QUANTITY = 1
GCM_NAME = 'EC-Earth3-Veg'
MAIN_SCENARIO = 'ssp585'


# # GENERAL VARIABLES
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']
predictands_map = {'ERA5-Land0.25deg': 'ERA5-Land', 'E-OBS': 'E-OBS','AEMET_0.25deg':'ROCIO-IBEB', 'Iberia01_v1.0':'Iberia01', 'CHELSA': 'CHELSA'}

hist_baseline = ('1995-01-01', '2014-12-31')
yearsTrain = ('1980-01-01', '2003-12-31')
yearsTest = ('2004-01-01', '2015-12-31')
future_1 = ('2021-01-01', '2040-12-31')
future_2 = ('2041-01-01', '2060-12-31')
future_3 = ('2081-01-01', '2100-12-31') 
future_4 = ('2061-01-01', '2080-12-31')


### # FIG1 # ####
if '1' in FIGS:
# DATOS OBSERVACION
    metrics = ['mean', '99quantile']
    obs = {}
    whole_obs = {'annual': {}}
    whole_obs_metrics = {'annual': {}}
    total_metrics = {f'{metric_stat}': [] for metric_stat in metrics}


    for predictand_name in predictands:

        modelName = f'DeepESD_tas_{predictand_name}' 

        obs[predictand_name] = utils.get_predictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
        obs[predictand_name] = obs[predictand_name].sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs[predictand_name] = utils.mask_data( # TODO Reemplazar a uso de d4d
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs[predictand_name],
                    secondGrid = obs[predictand_name])
        whole_obs['annual'][predictand_name] = obs[predictand_name]
        whole_obs_metrics['annual'][predictand_name] = utils.getMetricsTemp(whole_obs['annual'][predictand_name], short = True)
        for metric_stat in metrics:
            total_metrics[metric_stat].append(whole_obs_metrics['annual'][predictand_name][metric_stat])
    
    utils.create_multi_graph(
            data=whole_obs_metrics['annual'],
            vmin=[5, 15], vmax=[25, 35],
            fig_path=FIGS_PATH, fig_name=f'fig1_metrics_observation_annual_{yearsTrain[0]}-{yearsTest[1]}.pdf',
            n_rows=2, n_cols=len(whole_obs_metrics['annual']),
            cmap_colors=(0, 1, 11), cmap_first_color=1,
            cmap_min=0, cmap_max=6, tick_bool=False, 
            orientation=None,
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
            cmap_colors=(0, 1, 11), cmap_first_color=1,
            cmap_min=0, cmap_max=6, tick_bool=False, title='Std'
    )

            
    del total_metrics, total_metrics_concatened, std_metrics
    print("Figura 1 completada!")

### # FIG2 # ####
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
if '2' in FIGS:
    stat_metrics = ['rmse', 'bias', 'bias99']
    stats_map = {'rmse': 'RMSE', 'bias': 'Bias-mean', 'bias99': 'Bias-99th'}
    vminMetric = {'rmse': (0, 0, 20), 'bias': (0, 0, 20), 'bias99': (0.0, 0, 20)}
    vmaxMetric = {'rmse': (2.0, 20, 20), 'bias': (2, 20, 20), 'bias99': (2.0, 20, 20)}
    
    

    predictands_total_mean = []
    predictands_group_mean = []
    stat_realization_mean = {f'{metric}': {} for metric in stat_metrics}
    error_std = {f'{stat}': {'all': []} for stat in stat_metrics}
    for i, predictand_name in enumerate(predictands):      
        modelName = f'DeepESD_tas_{predictand_name}' 
        loaded_test_obs = utils.get_predictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
        loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
        loaded_test_obs = utils.mask_data(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = loaded_test_obs,
                    secondGrid = loaded_test_obs)
        
        predictand_numbered = [f"{predictand_name}_{i}" for i in range(1, ENSEMBLE_QUANTITY+1)]
        
        stat_realizations = {f'{stat}': [] for stat in stat_metrics}

        for predictand_number in predictand_numbered:
            modelName = f'deepesd_{predictand_number}_2004-01-01-2015-12-31'#f'DeepESD_tas_{predictand_number}' 
            loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}.nc')
            rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time']))['tasmean'])
            bias = np.abs(loaded_test.mean(['time'])['tasmean'] - loaded_test_obs.mean(['time'])['tasmean'])
            loaded_test_99 = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
            loaded_test_obs_99 = loaded_test_obs.resample(time = 'YE').quantile(0.99, dim = 'time')
            bias_99 = np.abs(loaded_test_99.mean(['time'])['tasmean'] - loaded_test_obs_99.mean(['time'])['tasmean'])

            stat_realizations['rmse'].append(rmse)
            stat_realizations['bias'].append(bias)
            stat_realizations['bias99'].append(bias_99)

        for stat_name, realization_list in stat_realizations.items():
            stat_concat = xr.concat(realization_list, dim='member')
            stat_realization_mean[stat_name][predictand_name] = stat_concat.mean(dim='member')

    figName = f'fig2_rmse_error_{ENSEMBLE_QUANTITY}'
    # Crear la figura y los ejes
    fig, axes = plt.subplots(3, 5, figsize=(20, 9), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
    inverted_stat_realization = {k: {i: d[k] for i, d in stat_realization_mean.items()} for k in next(iter(stat_realization_mean.values()))}

    utils.create_multi_graph(
            data = inverted_stat_realization,
            vmin=[0, 0, 0], vmax=[2, 2, 2],
            fig_path=FIGS_PATH, fig_name=figName, 
            n_rows=len(stat_metrics), n_cols=len(predictands),
            cmap_colors=(0, 1, 20),
            cmap_first_color=0, cmap_last_color=20,             
            color='viridis_r',
            cmap_min=0, cmap_max=6, tick_bool=False, 
            orientation='horizontal', spacing='uniform',
            var='tasmean', fontsize=15,
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
                data = stat_metrics,
                vmin=[0, 0, 0], vmax=[1, 1, 1],
                fig_path=FIGS_PATH, fig_name=f'fig2_std_error_{ENSEMBLE_QUANTITY}_{stat_name}', 
                n_rows=1, n_cols=1,
                cmap_colors=(0, 1, 11),
                cmap_first_color=1,
                color='cool',
                cmap_min=0, cmap_max=6, tick_bool=False, 
                orientation='horizontal', spacing='uniform',
                var='tasmean', fontsize=15,
                title=title
        )
    
        # discreteCMAPnoWhite2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, vmaxMetric[stat_name][2]+1)[1:]))
        # figName = '
        # # Crear la figura y los ejes
        # fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        # rmse_bias_mean_list = {key: list(value.values()) for key, value in stat_realization_mean.items()}
        # error_std[stat_name]['all'] = xr.concat(rmse_bias_mean_list[stat_name], dim='member').std(dim='member')   

        # data_to_plot = {'all-std': None}
        # data_to_plot['all-std'] = error_std[stat_name]['all']

        # ax1 = axes

        # if stat_name == 'rmse':
        #     ax1.set_title(f'Std', fontsize=16)

        # ax1.coastlines(resolution='10m')

        # im1 = ax1.pcolormesh(data_to_plot['all-std'].coords['lon'].values, data_to_plot['all-std'].coords['lat'].values,
        #                     data_to_plot['all-std'],
        #                     transform=ccrs.PlateCarree(),
        #                     cmap=discreteCMAPnoWhite2,
        #                     vmin=vminMetric[stat_name][0], vmax=vmaxMetric[stat_name][0])

        #position = [0.125, 0.055, 0.776, 0.065]
        # utils._add_colorbar(fig=fig, im=im, position=position,
        #                 vmin=vminMetric[stat_name][0],
        #                 vmax=vmaxMetric[stat_name][0],
        #                 cmap_max=6, fontsize=15,
        #                 tick_bool=False, orientation='horizontal')



        # #TODO plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.2, hspace=0.002)
        # plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        # plt.close()

    #del stat_realization_mean, error_std, data_to_plot, predictands_total_mean, predictands_group_mean, stat_realizations

    print("Figura 2 completada!")
### # FIG3 # ####
if '3' in FIGS:
    

    metric_label = {'Mean': 'Mean', '99Percentile': '99th'}
    position = [0.125, 0.510 - (j * 0.443), 0.776, 0.02]

    metrics = ['Mean', '99Percentile']
    # climatology - CCSIGNAL
    vminMetric = {'Mean': {'mean':4.6, 'std': 0.05, 'm-cmap':4, 'std-cmap':0}, 
                  '99Percentile': {'mean':5, 'std': 0.25, 'm-cmap':2, 'std-cmap':2}}
    vmaxMetric = {'Mean': {'mean':8.6, 'std': 0.85, 'm-cmap':14, 'std-cmap':8}, 
                  '99Percentile': {'mean':13, 'std': 1.25, 'm-cmap':22, 'std-cmap':12}}
    for metric in metrics:
        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part1'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
       

        predictands_total_mean = []
        predictands_group_mean = []

        for i, predictand_name in enumerate(predictands):
            # Historical Data
            obs_predictand = utils.get_predictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
            obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
            obs_predictand = utils.mask_data(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
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
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_data = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
                if metric == '99Percentile':
                    loaded_data = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
                mean_time = loaded_data.mean(dim='time')
                mean_list.append(mean_time)

            predictand_data_ensemble = xr.concat(mean_list, dim='member') - obs_predictand_mean

            predictand_data['mean'] = predictand_data_ensemble.mean('member')
            predictand_data['std'] = predictand_data_ensemble.std('member')
            predictands_total_mean.append(predictand_data['mean'])
            #TODO REVISAR ABAJO Y CREAR data_predictands con los datos de cada predictando de mean y std
        vminMetric = {'Mean': {'mean':4.6, 'std': 0.05, 'm-cmap':4, 'std-cmap':0}, 
                  '99Percentile': {'mean':5, 'std': 0.25, 'm-cmap':2, 'std-cmap':2}}
    vmaxMetric = {'Mean': {'mean':8.6, 'std': 0.85, 'm-cmap':14, 'std-cmap':8}, 
                  '99Percentile': {'mean':13, 'std': 1.25, 'm-cmap':22, 'std-cmap':12}}
        utils.create_multi_graph(
                data = stat_metrics,
                vmin=[vminMetric[metric]['mean'], vminMetric[metric]['std']], vmax=[vmaxMetric[metric]['mean'], vmaxMetric[metric]['std']],
                fig_path=FIGS_PATH, fig_name=figName, 
                n_rows=len(metrics), n_cols=len(data_predictands[metric]),
                cmap_colors=[(0, 1, 22), (0, 1, 11)],#REVISAR
                cmap_first_color=1, cmap_last_color=
                color='hot_r',
                cmap_min=0, cmap_max=6, tick_bool=False, 
                orientation='horizontal', spacing='uniform',
                var='tasmean', fontsize=16,
                title=title
        )
        #     for j, (metric_fig, metric_data) in enumerate(predictand_data.items()):
        #         if metric_fig == 'mean':
        #             vmin = vminMetric[metric]['mean']
        #             vmax = vmaxMetric[metric]['mean']
        #             num_ticks = 22
        #         elif metric_fig == 'std':
        #             vmin = vminMetric[metric]['std']
        #             vmax = vmaxMetric[metric]['std']
        #             num_ticks = 11

        #         ax = axes[j, i]
        #         if j == 0:
        #             ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
        #         if i == 0:
        #             metric_row = metric_label.get(metric, 'Std') if j==0 else 'Std'

        #             ax.text(-0.07, 0.55, f'{metric_row}', va='bottom', ha='center',
        #                 rotation='vertical', rotation_mode='anchor',
        #                 transform=ax.transAxes, fontsize=16)

        #         pos = 'm-cmap' if metric_fig=='mean' else 'std-cmap'
        #         cmap_min = vminMetric[metric][pos]
        #         cmap_max = vmaxMetric[metric][pos]
        #         continuousCMAP = plt.get_cmap('hot_r') if metric_fig == 'mean' else plt.get_cmap('cool')
        #         discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, num_ticks+1)[cmap_min:cmap_max]))

        #         ax.coastlines(resolution='10m')
        #         dataToPlot = metric_data['tasmean']
        #         im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
        #                             dataToPlot,
        #                             transform=ccrs.PlateCarree(),
        #                             cmap=discreteCMAPnoWhite,
        #                             vmin=vmin, vmax=vmax)

        #         if i == 0:
        #             utils._add_colorbar(fig, im, position, vmin, vmax, cmap_min, cmap_max+1, fontsize=16)
                    

        # plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        # plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        # plt.close()

        data = {}
        continuousCMAP2 = plt.get_cmap('cool')
        num_ticks = 11
        discreteCMAP2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, num_ticks)[vminMetric[metric][3]:vmaxMetric[metric][3]]))
        discreteCMAPnoWhite2 = ListedColormap(continuousCMAP2(np.linspace(0, 1, num_ticks+1)[vminMetric[metric][3]:vmaxMetric[metric][3]]))

        figName = f'fig3_Statistics_CCSignal_{ENSEMBLE_QUANTITY}_{metric}_part2'
        # Crear la figura y los ejes
        fig, axes = plt.subplots(1, 1, figsize=(4, 3), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

        data_to_plot = {'all-std': None}
        mean_combined = xr.concat(predictands_total_mean, dim='member')
        data_to_plot['all-std'] = mean_combined.std(dim='member')

        vmin = vminMetric[metric][1]
        vmax = vmaxMetric[metric][1]

        ax1 = axes
        ax1.set_title(f'Std', fontsize=16)
        ax1.coastlines(resolution='10m')

        im1 = ax1.pcolormesh(data_to_plot['all-std']['tasmean'].coords['lon'].values, data_to_plot['all-std']['tasmean'].coords['lat'].values,
                            data_to_plot['all-std']['tasmean'],
                            transform=ccrs.PlateCarree(),
                            cmap=discreteCMAPnoWhite2,
                            vmin=vmin, vmax=vmax)
        
        utils._add_colorbar(fig, im, position, vmin, vmax, fontsize=16)

        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()

    del predictand_data, predictand_data_ensemble, mean_combined, predictands_total_mean, predictands_group_mean, mean_list

    print("Figura 3 completada!")



### # FIG 4 # ####
if '4' in FIGS:
    shape_name_list = ['Iberia', 'Pirineos', 'Tinto', 'Duero']
    #*********************************************************************+
    references_grid = {shape: [] for shape in shape_name_list}
    reference_grid = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_DeepESD_tas_AEMET_0.25deg_1_{GCM_NAME}_{MAIN_SCENARIO}_{future_1[0]}-{future_1[1]}.nc')
    reference_grid = reference_grid.sel(time=slice(future_1[0],'2021-01-02'))
    for shape in shape_name_list:
        if shape == 'Iberia':
            references_grid[shape] = None
        elif shape == 'Pirineos':
            references_grid[shape] = reference_grid.sel(lons=(-0.37, 3.37), lats=(41.42, 42.80))
        elif shape == 'Duero':
            references_grid[shape] = reference_grid.sel(lons=(-6.59, -4.75), lats=(40.85, 42.45))
        elif shape == 'Tinto':
            references_grid[shape] = reference_grid.sel(lons=(-7.23, -5.20), lats=(36.00, 38.20))


    # GRAFICOS BOXPLOT PARA SHORT, MEDIUM y LONG / CCSIGNAL
    periods = [future_2, future_4, future_3]
    xmin = (2, 1.5)
    xmax = (11, 11.5)
    # CC SIGNAL
    for shape in shape_name_list:
        # Etiquetas
        colors = ['darkgreen', 'darkblue', 'darkred']
        names = ['Short', 'Medium', 'Long']
        legend_handles = []

        figName = f'fig4_boxPlot_ccsignals_Ensemble{ENSEMBLE_QUANTITY}_{shape}'
        # Crear la figura y los ejes
        fig, ax1 = plt.subplots(figsize=(20, 12))

        # Graficar cada set de datos (Short, Medium, Long) en el mismo gráfico
        for i, period in enumerate(periods):
            data_to_plot = []
            data_to_plot_99 = []
            for predictand_name in predictands:
                obs2 = utils.get_predictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
                obs_temp = obs2.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
                obs2 = utils.mask_data(
                            path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                            var='tasmean',
                            to_slice=(hist_baseline[0], hist_baseline[1]),
                            objective = obs2.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
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
                    modelName = f'DeepESD_tas_{predictand_number}' 
                    loaded_data = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{period[0]}-{period[1]}.nc')
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

            ax = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            color = colors[i]
            bplot = ax.boxplot(data_to_plot, positions= 5 + np.arange(len(predictands)), widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=2))
            ax.set_xlim(xmin[1], xmax[1]-2)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.tick_top()
            ax.set_xticks(np.linspace(xmin[0], xmax[0]-2, 8))
            ax.tick_params(axis='x', labelsize=16)
 

            ax_99 = ax1.twiny() if i > 0 else ax1  # Crear ejes adicionales solo para Medium y Long
            bplot = ax_99.boxplot(data_to_plot_99, positions=np.arange(len(predictands))+0.1 , widths=0.35, 
                            patch_artist=True, boxprops=dict(facecolor=color), vert=False, whis=[5, 95],
                            whiskerprops=dict(color=color), flierprops=dict(color=color, markeredgecolor=color),
                            medianprops=dict(color='snow', linewidth=1.5))
            ax_99.set_xlim(xmin[1], xmax[1])
            ax_99.xaxis.set_ticks_position('bottom')
            ax_99.set_xticks([]) if i>0 else ax_99.set_xticks(np.linspace(xmin[0], xmax[0], 10))
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
    figName = f'fig5_extremes_ccsignals_Ensemble{ENSEMBLE_QUANTITY}'
    # fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharex=False, sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})

    # continuousCMAP = plt.get_cmap('hot_r')
    # discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[1:]))
    graph_data = {}

    for i, predictand_name in enumerate(predictands):
        # Observed Data
        obs_predictand = utils.get_predictand(f'{DATA_PATH_PREDICTANDS_SAVE}', predictand_name, 'tasmean')
        obs_temp = obs_predictand.sel(time=slice(*(yearsTrain[0], yearsTest[1])))
        obs_predictand = utils.mask_data(
                    path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
                    var='tasmean',
                    to_slice=(yearsTrain[0], yearsTest[1]),
                    objective = obs_predictand.sel(time=slice(*(hist_baseline[0], hist_baseline[1]))),
                    secondGrid = obs_temp)

        obs_predictand_mean = obs_predictand.mean(dim='time')
        obs_predictand_mean_99 = obs_predictand.resample(time = 'YE').quantile(0.99, dim = 'time').mean(dim='time')


        # Future Data
        predictand_data = {}
        predictand_data_mean = {}
        number_min_max = {}

        for p_num in range(1, ENSEMBLE_QUANTITY+1):
            
            modelName = f'DeepESD_tas_{predictand_name}_{p_num}' 
            loaded_data = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
            

            grided_mean = loaded_data.mean(dim=['time', 'lat', 'lon']) 
            mean_time = loaded_data.mean(dim='time')

            loaded_data_99 = loaded_data.resample(time = 'YE').quantile(0.99, dim = 'time')
            grided_mean_99 = loaded_data_99.mean(dim=['time', 'lat', 'lon']) 
            mean_time_99 = loaded_data_99.mean(dim='time')


            # CHECK MIN AND MAX AND SAVE MIN, MAX, MEAN FOR CCSIGNAL
            for key, value in {'min_mean': (mean_time, grided_mean['tasmean'].values),
                            'max_mean': (mean_time, grided_mean['tasmean'].values),
                            'min_99': (mean_time_99, grided_mean_99['tasmean'].values),
                            'max_99': (mean_time_99, grided_mean_99['tasmean'].values)}:
                if predictand_data_mean[key] == None or value[1] < predictand_data_mean[key]:
                    predictand_data_mean[key] = value[1]
                    predictand_data[key] = value[0]
                    number_min_max[key] = p_num

            
        graph_data[predictand_name] = {
            'Rmax-Rmin (Mean)': predictand_data['max_mean'] - predictand_data['min_mean'],
            'Rmax-Rmin (99th)': predictand_data['max_99'] - predictand_data['min_99'],
        }

    # GRAFICOS
    utils.create_multi_graph(
            data = stat_metrics,
            vmin=[0, 0], vmax=[2, 4],
            fig_path=FIGS_PATH, fig_name=figName, 
            n_rows=2, n_cols=len(graph_data),
            cmap_colors=(0, 1, 11),
            cmap_first_color=[1, 1], cmap_last_color=[6, 11]
            color='hot_r',
            cmap_min=0, cmap_max=6, tick_bool=False, 
            orientation='horizontal', spacing='uniform',
            var='tasmean', fontsize=15,
            x_map=predictands_map, y_map={'Rmax-Rmin (Mean)': 'Rmax-Rmin (Mean)', 'Rmax-Rmin (99th)': 'Rmax-Rmin (99th)'}
    )
    #     vmax = {'Rmax-Rmin (Mean)': 2, 'Rmax-Rmin (99th)': 4}
    #     vmin = {'Rmax-Rmin (Mean)': 0, 'Rmax-Rmin (99th)': 0}

    #     colors = {'Rmax-Rmin (Mean)': (1, 6), 'Rmax-Rmin (99th)': (1, 11)}
    #     for j, (metric, metric_data) in enumerate(predictand_data_final.items()):
            
    #         discreteCMAPnoWhite = ListedColormap(continuousCMAP(np.linspace(0, 1, 11)[colors[metric][0]:colors[metric][1]]))
    #         ax = axes[j, i]
    #         if j == 0:
    #             ax.set_title(f'{predictands_map[predictand_name]}', fontsize=16)
    #         if i == 0:
    #             ax.text(-0.07, 0.55, f'{metric}', va='bottom', ha='center',
    #                 rotation='vertical', rotation_mode='anchor',
    #                 transform=ax.transAxes, fontsize=16)

    #         ax.coastlines(resolution='10m')
            

    #         dataToPlot = metric_data['tasmean']
    #         im = ax.pcolormesh(dataToPlot.coords['lon'].values, dataToPlot.coords['lat'].values,
    #                             dataToPlot,
    #                             transform=ccrs.PlateCarree(),
    #                             cmap=discreteCMAPnoWhite,
    #                             vmin=vmin[metric], vmax=vmax[metric])
            
    #         if i == 0:
    #             position = [0.125, 0.510 - (j * 0.443), 0.776, 0.02]
    #             utils._add_colorbar(fig=fig, im=im, position=position,
    #                     vmin=vmin[metric],
    #                     vmax=vmax[metric],
    #                     cmap_max=6,
    #                     orientation='horizontal')


    # plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
    # plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
    # plt.close()

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

            loaded_test_obs = utils.get_predictand(DATA_PATH_PREDICTANDS_SAVE, predictand_name, 'tasmean')
            loaded_test_obs = loaded_test_obs.sel(time=slice(*(yearsTest[0], yearsTest[1])))
            loaded_test_obs = utils.mask_data(
                        path = f'{DATA_PATH_PREDICTANDS_SAVE}AEMET_0.25deg/AEMET_0.25deg_tasmean_1951-2022.nc',
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
                modelName = f'DeepESD_tas_{predictand_number}' 
                loaded_test = xr.open_dataset(f'{PREDS_PATH_TEST}predTest_{modelName}.nc')
                rmse = np.sqrt((((loaded_test - loaded_test_obs)**2).mean(dim=['time', 'lat', 'lon']))['tasmean'])
                rmse_test.append(rmse)

                if metric == '99Percentile':
                    loaded_test = loaded_test.resample(time = 'YE').quantile(0.99, dim = 'time')
                test_pred.append(loaded_test.mean(dim=['time', 'lat', 'lon'])['tasmean'])

                
                loaded_pred = xr.open_dataset(f'{PREDS_PATH_GCM}/predGCM_{modelName}_{GCM_NAME}_{MAIN_SCENARIO}_{future_3[0]}-{future_3[1]}.nc')
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
        ax.set_xlim(valuesMinMax[metric][0], valuesMinMax[metric][3])#ax.set_xlim(np.floor(min([da.values.item() for da in gcm_pred])), np.ceil(max([da.values.item() for da in gcm_pred])))
        ax.set_ylim(valuesMinMax[metric][0], valuesMinMax[metric][3])#ax.set_ylim(np.floor(min([da.values.item() for da in test_pred])), np.ceil(max([da.values.item() for da in test_pred])))
        ax.set_xlabel('Temperature Test', fontsize=12)
        ax.set_ylabel('Temperature Long', fontsize=12)
        ax.set_title('Long vs Test', fontsize=14)
        ax.legend()

        # Show grid for better readability
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.subplots_adjust(top=0.95, bottom=0.05, wspace=0.002, hspace=0.002)
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
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
        plt.savefig(f'{FIGS_PATH}{figName}.png', bbox_inches='tight')
        plt.savefig(f'{FIGS_PATH}{figName}.pdf', bbox_inches='tight')
        plt.close()


    print("Figura 6 completada!")

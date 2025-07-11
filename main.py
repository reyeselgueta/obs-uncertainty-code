#!git clone https://github.com/SantanderMetGroup/deep4downscaling.git


import os
import time
import numpy as np
import sys
from pathlib import Path
import xarray as xr
import torch
from torch.utils.data import DataLoader, random_split

BASE_PATH = Path("/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling")#"/vols/abedul/home/meteo/reyess/paper1-code/deep4downscaling")
sys.path.insert(0, str(BASE_PATH))

import deep4downscaling.trans as deep_trans
import deep4downscaling.deep.loss as deep_loss
import deep4downscaling.deep.utils as deep_utils
import deep4downscaling.deep.models as deep_models
import deep4downscaling.deep.train as deep_train
import deep4downscaling.deep.pred as deep_pred




DATA_PATH_PREDICTAND = '/lustre/gmeteo/WORK/reyess/data/predictand/'#./data/input'
DATA_PATH_PREDICTOR = '/lustre/gmeteo/WORK/reyess/data/NorthAtlanticRegion_1.5degree/'
DATA_PATH = '/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/data/input'
FIGURES_PATH = '/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/figures'
MODELS_PATH = '/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/models'
ASYM_PATH = '/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/data/asym'
predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']

gcm_name = 'EC-Earth3-Veg'
main_scenario = 'ssp585'
ensemble_start = int(sys.argv[2])
ensemble_quantity = int(sys.argv[3])
predictand_to_train = sys.argv[1]
predictands_to_train = predictands


predictors_vars = ['t500', 't700', 't850', # Air temperature at 500, 700, 850 hPa
'q500', 'q700', 'q850', # Specific humidity at 500, 700, 850 hPa
'v500', 'v700', 'v850', # Meridional wind component at 500, 700, 850 hPa
'u500', 'u700', 'u850', # Zonal wind component at 500, 700, 850 hPa
'msl'] # Mean sea level pressure (psl)


VARIABLES_TO_DROP = ['lon_bnds', 'lat_bnds', 'crs']


predictors_filename = os.path.join(DATA_PATH_PREDICTOR, "*ERA5.nc")

predictor = xr.open_mfdataset(
    predictors_filename,
    combine="by_coords"
)

# Remove days with nans in the predictor
predictor = deep_trans.remove_days_with_nans(predictor)

# Años a entranar y test
years_train = ('1980-01-01', '2003-12-31')#2003
years_test = ('2004-01-01', '2015-12-31')
# Fechas a eliminar
fechas_a_eliminar = ["1982-02-28", "1986-02-28", "1990-02-28", "1994-02-28", "1998-02-28", "2002-02-28", "2006-02-28", "2010-02-28", "2014-02-28"]
fechas_a_eliminar = np.array(fechas_a_eliminar, dtype="datetime64")

predictand_dict = {}
predictand_dates = {'ERA5-Land0.25deg': 'ERA5-Land0.25deg_tasmean_1971-2022.nc',
                    'AEMET_0.25deg': 'AEMET_0.25deg_tasmean_1951-2022.nc',
                    'E-OBS': 'tasmean_e-obs_v27e_0.10regular_Spain_0.25deg_reg_1950-2022.nc', 
                    'Iberia01_v1.0': 'Iberia01_v1.0_tasmean_1971-2015.nc',
                    'CHELSA': 'CHELSA_tasmean_1979-2016.nc'}#TODO Borrar
for predictand_name in predictands:
    predictand_filename = f'{DATA_PATH_PREDICTAND}/{predictand_name}/{predictand_dates[predictand_name]}'#TODO Rellenar por usuario
    predictand = xr.open_dataset(predictand_filename)
    predictand["time"] = predictand["time"].dt.floor("D")
    predictand = predictand.sel(time=slice(years_train[0], years_test[1]))
    ##### ESTA PARTE HACERLA EN utils como antes
    is_kelvin = (predictand['tasmean'] > 100).any(dim=('time', 'lat', 'lon'))
    if is_kelvin:
        predictand['tasmean'].data = predictand['tasmean'].data - 273.15
    if predictand_name == 'CHELSA':
        mask = (predictand['tasmean'] < -200).any(dim=['lat', 'lon'])
        mask_computed = mask.compute()
        wrong_days = predictand['time'].where(mask_computed, drop=True)
        data_cleaned = predictand.drop_sel(time=wrong_days.values)

    # Align both datasets in time
    predictor, predictand = deep_trans.align_datasets(predictor, predictand, 'time')
    predictand_dict[predictand_name] = predictand


predictand_merge_train = xr.concat(predictand_dict.values(), dim='stacked').sum(dim='stacked', skipna=False)

y_mask = deep_trans.compute_valid_mask(predictand_merge_train) 


num_epochs = 10000
patience_early_stopping = 20
learning_rate = 0.0001
device = ('cuda' if torch.cuda.is_available() else 'cpu')
loss_function = deep_loss.MseLoss(ignore_nans=True)

# Preparacion de predictor
x_train = predictor.sel(time=slice(*years_train))
x_test = predictor.sel(time=slice(*years_test))
x_train = x_train.sel(time=~x_train.time.isin(fechas_a_eliminar))
x_test = x_test.sel(time=~x_test.time.isin(fechas_a_eliminar))
x_train_stand = deep_trans.standardize(data_ref=x_train, data=x_train)
x_train_stand_arr = deep_trans.xarray_to_numpy(x_train_stand)

# ENTRENAMIENTO
for predictand_name in predictands_to_train:

    
    y_train = predictand_dict[predictand_name].sel(time=slice(*years_train))
    y_test = predictand_dict[predictand_name].sel(time=slice(*years_test))
    y_train = y_train.sel(time=~y_train.time.isin(fechas_a_eliminar))
    y_test = y_test.sel(time=~y_test.time.isin(fechas_a_eliminar))

    y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
    y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

    y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
    y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                                drop=True)

    y_train_arr = deep_trans.xarray_to_numpy(y_train_stack_filt)

    # Create Dataset
    train_dataset = deep_utils.StandardDataset(x=x_train_stand_arr,
                                                                y=y_train_arr)

    

    for num in range(ensemble_start, ensemble_quantity):
        # Split into training and validation sets
        train_dataset, valid_dataset = random_split(train_dataset,
                                                    [0.9, 0.1])

        # Create DataLoaders
        batch_size = 64

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                                    shuffle=True)

        model_name = f'deepesd_{predictand_name}_{num}'
        print(f"Comienza el modelo: {model_name}")

        model = deep_models.DeepESDtas(
            x_shape=x_train_stand_arr.shape,
            y_shape=y_train_arr.shape,
            filters_last_conv=10,
            stochastic=False
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print(f'Model {num}, weight norm:', model.conv_1.weight.norm().item())
        train_loss, val_loss = deep_train.standard_training_loop(
                                    model=model, model_name=model_name, model_path=MODELS_PATH,
                                    device=device, num_epochs=num_epochs,
                                    loss_function=loss_function, optimizer=optimizer,
                                    train_data=train_dataloader, valid_data=valid_dataloader,
                                    patience_early_stopping=patience_early_stopping)
        
        # PREDICTIONS
        model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt'))

        # Standardize
        x_test_stand = deep_trans.standardize(data_ref=x_train, data=x_test)

        # Compute predictions
        pred_test = deep_pred.compute_preds_standard(
                                        x_data=x_test_stand, model=model,
                                        device=device, var_target='tasmean',
                                        mask=y_mask, batch_size=16)


        pred_test.to_netcdf(f'/oceano/gmeteo/users/reyess/paper1-code/preds/test/predTest_{model_name}_{years_test[0]}-{years_test[1]}.nc')#TODO ./preds/test


years_gcm = ('2081', '2100')
years_reference = ('1980-01-01', '2014-12-31')
era5_reference = predictor.sel(time=slice(*years_reference))
gcm_hist = xr.open_dataset(f'/oceano/gmeteo/users/reyess/d4d-fork/deep4downscaling/notebooks/data/input/EC-Earth3-Veg_historical_r1i1p1f1_19500101-20141231.nc')#TODO ./
gcm_predictor = xr.open_dataset(f'/oceano/gmeteo/users/reyess/d4d-fork/deep4downscaling/notebooks/data/input/EC-Earth3-Veg_ssp585_r1i1p1f1_20150101-21001231.nc')#TODO ./
gcm_predictor = gcm_predictor.sel(time=slice(*years_gcm))
gcm_fut_corrected = deep_trans.scaling_delta_correction(data=gcm_predictor,
                                                                    gcm_hist=gcm_hist, obs_hist=era5_reference.load())
gcm_fut_corrected_stand = deep_trans.standardize(data_ref=x_train.load(), data=gcm_fut_corrected)



for predictand_name in predictands_to_train:
    y_train = predictand_dict[predictand_name].sel(time=slice(*years_train))
    y_train = y_train.sel(time=~y_train.time.isin(fechas_a_eliminar))
    y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
    y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

    y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
    y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                                drop=True)

    y_train_arr = deep_trans.xarray_to_numpy(y_train_stack_filt)
    model = deep_models.DeepESDtas(x_shape=x_train_stand_arr.shape,
                                                y_shape=y_train_arr.shape,
                                                filters_last_conv=10,
                                                stochastic=False)
    for num in range(ensemble_start, ensemble_quantity):
        model_name = f'deepesd_{predictand_name}_{num}'
        model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt'))

        # Compute predictions
        proj_future = deep_pred.compute_preds_standard(
                    x_data=gcm_fut_corrected_stand, model=model,
                    device=device, var_target='tasmean',
                    mask=y_mask, batch_size=16)


        proj_future.to_netcdf(f'/oceano/gmeteo/users/reyess/paper1-code/preds/gcm/predGCM_{model_name}_{gcm_name}_{main_scenario}_{years_gcm[0]}-{years_gcm[1]}.nc')

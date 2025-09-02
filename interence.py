import os
import time
import sys
import xarray as xr
import numpy as np
import utils
import torch
from pathlib import Path

BASE_PATH = Path("/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling")#"/vols/abedul/home/meteo/reyess/paper1-code/deep4downscaling")
sys.path.insert(0, str(BASE_PATH))

import deep4downscaling.trans as deep_trans
import deep4downscaling.deep.models as deep_models
import deep4downscaling.deep.pred as deep_pred


DATA_PATH_PREDICTAND = '...'
DATA_PATH_PREDICTOR = '...'
DATA_PATH = './notebooks/data/input/'
FIGURES_PATH = './notebooks/figures/'
MODELS_PATH = './notebooks/models/'
ASYM_PATH = './notebooks/data/asym/'
PREDS_PATH = './preds/'

predictand_name = sys.argv[1]
num = int(sys.argv[2])


start = time.time() 

predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']
gcm_name = 'EC-Earth3-Veg'
main_scenario = 'ssp585'



predictors_filename = os.path.join(DATA_PATH_PREDICTOR, "*ERA5.nc")#DATA_PATH_PREDICTOR

predictor = xr.open_mfdataset(
    predictors_filename,
    combine="by_coords"
)
predictor = predictor.reindex(lat=list(reversed(predictor.lat))) 

#En caso de entrenar varios modelos a la vez usar este linea ara evitar cuello de botella en disco duro.
predictor = predictor.load()

# Remove days with nans in the predictor
predictor = deep_trans.remove_days_with_nans(predictor)

# Años a entranar y test
years_train = ('1980-01-01', '2003-12-31')#2003
years_test = ('2004-01-01', '2015-12-31')
# Fechas a eliminar (Debido a que dataset de CHELSA no tiene datos esos dias)
fechas_a_eliminar = ["1982-02-28", "1986-02-28", "1990-02-28", "1994-02-28", "1998-02-28", "2002-02-28", "2006-02-28", "2010-02-28", "2014-02-28"]
fechas_a_eliminar = np.array(fechas_a_eliminar, dtype="datetime64")

predictand_dict = {}
predictand_dates = {'ERA5-Land0.25deg': 'ERA5-Land0.25deg_tasmean_1971-2022.nc',
                    'AEMET_0.25deg': 'AEMET_0.25deg_tasmean_1951-2022.nc',
                    'E-OBS': 'tasmean_e-obs_v27e_0.10regular_Spain_0.25deg_reg_1950-2022.nc', 
                    'Iberia01_v1.0': 'Iberia01_v1.0_tasmean_1971-2015.nc',
                    'CHELSA': 'CHELSA_tasmean_1979-2016.nc'}#TODO Borrar

for name in predictands:
    predictand_filename = f'{DATA_PATH_PREDICTAND}/{name}/{predictand_dates[name]}'#TODO Rellenar por usuario
    predictand = xr.open_dataset(predictand_filename)
    predictand["time"] = predictand["time"].dt.floor("D")
    predictand = predictand.sel(time=slice(years_train[0], years_test[1]))
    predictand = predictand.load() # Solo si se entrena varios modelos a la vez

    # Descartar dias erroneos
    predictand = predictand.sel(time=~predictand.time.isin(fechas_a_eliminar))
    # Transformar valores de kelvin a celcius
    is_kelvin = (predictand['tasmean'] > 100).any(dim=('time', 'lat', 'lon'))
    if is_kelvin:
        predictand['tasmean'].data = predictand['tasmean'].data - 273.15
    # Descatar dias con valores en zero absoluto de CHELSA
    if name == 'CHELSA':
        mask = (predictand['tasmean'] < -200).any(dim=['lat', 'lon'])
        mask_computed = mask.compute()
        wrong_days = predictand['time'].where(mask_computed, drop=True)
        predictand = predictand.drop_sel(time=wrong_days.values)

    # Align both datasets in time
    predictor, predictand = deep_trans.align_datasets(predictor, predictand, 'time')
    predictand_dict[name] = predictand


predictand_merge_train = xr.concat(predictand_dict.values(), dim='stacked').mean(dim='stacked', skipna=False)

y_mask = deep_trans.compute_valid_mask(predictand_merge_train) 


# Preparacion de predictor
x_train = predictor.sel(time=slice(*years_train))
x_test = predictor.sel(time=slice(*years_test))
x_train_stand = deep_trans.standardize(data_ref=x_train, data=x_train)
x_train_stand_arr = deep_trans.xarray_to_numpy(x_train_stand)

# Preparacion de predictando
y_train = predictand_dict[predictand_name].sel(time=slice(*years_train))
y_test = predictand_dict[predictand_name].sel(time=slice(*years_test))

y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                            drop=True)
y_train_arr = deep_trans.xarray_to_numpy(y_train_stack_filt)

utils.set_seed(num)
model_name = f'deepesd_{predictand_name}_{num}'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cpu")
print(f"Comienza el modelo: {model_name}")

model = deep_models.DeepESDtas(
    x_shape=x_train_stand_arr.shape,
    y_shape=y_train_arr.shape,
    filters_last_conv=10,
    stochastic=False
)
print(device)
# Cargar los pesos en esa instancia
state_dict = torch.load(f"{MODELS_PATH}/{model_name}.pt", map_location=device)
model.load_state_dict(state_dict, strict=True)

# Standardize
x_test_stand = deep_trans.standardize(data_ref=x_train, data=x_test)

# Compute predictions
pred_test = deep_pred.compute_preds_standard(
                                x_data=x_test_stand, model=model,
                                device=device, var_target='tasmean',
                                mask=y_mask, batch_size=16)


pred_test.to_netcdf(f'{PREDS_PATH}test/predTest_{model_name}_{years_test[0]}-{years_test[1]}.nc')#TODO ./preds/test
        


# GCM Predictions
years_reference = ('1980-01-01', '2014-12-31')
gcm_hist = xr.open_dataset(f'/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/data/input/EC-Earth3-Veg_r1i1p1f1_hist.nc')
gcm_predictor = xr.open_dataset(f'/oceano/gmeteo/users/reyess/paper1-code/deep4downscaling/notebooks/data/input/EC-Earth3-Veg_r1i1p1f1_ssp585_fut.nc')#TODO ./

gcm_hist = gcm_hist.sel(time=slice(*years_reference))
era5_reference = predictor.sel(time=slice(*years_reference))

for years_gcm in [('2021', '2040'), ('2041', '2060'), ('2061', '2080'), ('2081', '2100')]:
    
    gcm_predictor_sliced = gcm_predictor.sel(time=slice(*years_gcm))

    #Ajustes al gcm
    gcm_predictor_sorted = deep_trans.sort_variables(data=gcm_predictor_sliced, ref=era5_reference, keep_vars=False)
    
    gcm_fut_corrected = deep_trans.scaling_delta_correction(data=gcm_predictor_sorted,
                                                                        gcm_hist=gcm_hist, obs_hist=era5_reference)
    gcm_fut_corrected_stand = deep_trans.standardize(data_ref=x_train, data=gcm_fut_corrected)



    #for predictand_name in predictands_to_train:
    y_train = predictand_dict[predictand_name].sel(time=slice(*years_train))
    y_train = y_train.sel(time=~y_train.time.isin(fechas_a_eliminar))
    y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
    y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

    y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
    y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                                drop=True)

    y_train_arr = deep_trans.xarray_to_numpy(y_train_stack_filt)
    
    #for num in range(ensemble_start, ensemble_quantity):
    model = deep_models.DeepESDtas(x_shape=x_train_stand_arr.shape,
                                            y_shape=y_train_arr.shape,
                                            filters_last_conv=10,
                                            stochastic=False)
    model_name = f'deepesd_{predictand_name}_{num}'
    state_dict = torch.load(f"{MODELS_PATH}/{model_name}.pt", map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Compute predictions
    proj_future = deep_pred.compute_preds_standard(
                x_data=gcm_fut_corrected_stand, model=model,
                device=device, var_target='tasmean',
                mask=y_mask, batch_size=16)

    proj_future.to_netcdf(f'{PREDS_PATH}gcm/predGCM_{model_name}_{gcm_name}_{main_scenario}_{years_gcm[0]}-{years_gcm[1]}.nc')

    del model


end = time.time() 
elapsed_minutes = (end - start) / 60

print(f"Tiempo transcurrido: {elapsed_minutes:.2f} minutos")

print("Test and GCM Inferences Finished")

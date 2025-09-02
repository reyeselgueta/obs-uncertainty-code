!git clone https://github.com/SantanderMetGroup/deep4downscaling.git

import os
import time
import numpy as np
import sys
import utils
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
import deep4downscaling.viz as deep_viz



DATA_PATH_PREDICTAND = '...'
DATA_PATH_PREDICTOR = '...'
DATA_PATH = './notebooks/data/input/'
FIGURES_PATH = './notebooks/figures/'
MODELS_PATH = './notebooks/models/'
ASYM_PATH = './notebooks/data/asym/'
PREDS_PATH = './preds/'


predictand_name = sys.argv[1]
num = int(sys.argv[2])

predictands = ['ERA5-Land0.25deg', 'E-OBS','AEMET_0.25deg', 'Iberia01_v1.0', 'CHELSA']
gcm_name = 'EC-Earth3-Veg'
main_scenario = 'ssp585'

start = time.time()

predictors_filename = os.path.join(f'{DATA_PATH_PREDICTOR}NorthAtlanticRegion_1.5degree/', "*ERA5.nc")#DATA_PATH_PREDICTOR

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
years_test = ('2004-01-01', '2015-12-31')#2015
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


num_epochs = 10000
patience_early_stopping = 30
learning_rate = 0.0001
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE")
print(device)
loss_function = deep_loss.MseLoss(ignore_nans=True)#False

# Preparacion de predictor
x_train = predictor.sel(time=slice(*years_train))
x_test = predictor.sel(time=slice(*years_test))
x_train_stand = deep_trans.standardize(data_ref=x_train, data=x_train)
x_train_stand_arr = deep_trans.xarray_to_numpy(x_train_stand)

# ENTRENAMIENTO
utils.set_seed(num)
y_train = predictand_dict[predictand_name].sel(time=slice(*years_train))
y_test = predictand_dict[predictand_name].sel(time=slice(*years_test))

y_train_stack = y_train.stack(gridpoint=('lat', 'lon'))
y_mask_stack = y_mask.stack(gridpoint=('lat', 'lon'))

y_mask_stack_filt = y_mask_stack.where(y_mask_stack==1, drop=True)
y_train_stack_filt = y_train_stack.where(y_train_stack['gridpoint'] == y_mask_stack_filt['gridpoint'],
                                            drop=True)

y_train_arr = deep_trans.xarray_to_numpy(y_train_stack_filt)

# Create Dataset
train_dataset = deep_utils.StandardDataset(x=x_train_stand_arr,
                                                            y=y_train_arr)

    

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

train_loss, val_loss = deep_train.standard_training_loop(
                            model=model, model_name=model_name, model_path=MODELS_PATH,
                            device=device, num_epochs=num_epochs,
                            loss_function=loss_function, optimizer=optimizer,
                            train_data=train_dataloader, valid_data=valid_dataloader,
                            patience_early_stopping=patience_early_stopping)

model.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}.pt'))


end = time.time() 
elapsed_minutes = (end - start) / 60

print(f"Tiempo transcurrido: {elapsed_minutes:.2f} minutos")

print("Model Train Finished")

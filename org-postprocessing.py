#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Custom_loss_original import MaskedWeightedMAEPrecip

print("¶ Starting full Prism postprocessing...")

# -----------------------------------------------------------------------------
# 1. Load normalization stats
# -----------------------------------------------------------------------------
mean_pr = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/ERA5_mean_train.npy', allow_pickle=True)
std_pr  = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/ERA5_std_train.npy',  allow_pickle=True)
# if they came in as (lat,lon,1) squeeze to (lat,lon)
if mean_pr.ndim == 3 and mean_pr.shape[-1] == 1:
    mean_pr = mean_pr[...,0]
    std_pr  = std_pr[...,0]
H, W = mean_pr.shape
print(f"  mean_pr / std_pr shape = {mean_pr.shape}")

# -----------------------------------------------------------------------------
# 1b. Locate & sanitize your 2D land mask
# -----------------------------------------------------------------------------
mask_candidates = ['land_mask.npy', '/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/land_mask.npy']
land_mask = None
for mp in mask_candidates:
    if os.path.isfile(mp):
        land_mask = np.load(mp, allow_pickle=True)
        print(f"  Loaded land_mask from {mp}, raw shape = {land_mask.shape}")
        break
if land_mask is None:
    raise FileNotFoundError("Could not find a 2D land_mask.npy in the project or mydata/")

# if it s accidentally got a time axis first (T,H,W), strip that off:
if land_mask.ndim == 3 and land_mask.shape[1:] == (H, W):
    print("  land_mask has time dimension; taking mask[0] ’ spatial only")
    land_mask = land_mask[0]
if land_mask.shape != (H, W):
    raise ValueError(f"land_mask must be 2D {(H,W)}, but got {land_mask.shape}")

# build the masked, weighted MAE loss
loss_fn = MaskedWeightedMAEPrecip(
    mean_pr, std_pr, land_mask,
    scale=36.0, w_min=0.2, w_max=1.82
)

# -----------------------------------------------------------------------------
# 2. Load predictors & observations (normalized log1p), train + test
# -----------------------------------------------------------------------------
X_tr = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/predictors_train_mean_std_separate.npy')
X_te = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/predictors_test_mean_std_separate.npy')
y_tr = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/obs_train_mean_std_single.npy')
y_te = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/obs_test_mean_std_single.npy')

n_tr, n_te = X_tr.shape[0], X_te.shape[0]
print(f"  n_train={n_tr}, n_test={n_te}")

# stack them
X_all = np.concatenate([X_tr, X_te], axis=0)
y_all = np.concatenate([y_tr, y_te], axis=0)
# ensure y_all is 4D: (batch, H, W, 1)
if y_all.ndim == 3:
    y_all = y_all[...,None]
print(f"  X_all.shape = {X_all.shape}")
print(f"  y_all.shape = {y_all.shape}")

# -----------------------------------------------------------------------------
# 3. Load generator & predict (normalized log1p)
# -----------------------------------------------------------------------------
model = load_model(
    'save_model/generator_160.h5',
    custom_objects={'MaskedWeightedMAEPrecip': MaskedWeightedMAEPrecip}
)
print("  Model loaded; running .predict()&")
y_pred_normlog = model.predict(X_all, verbose=1)
print(f"  y_pred_normlog.shape = {y_pred_normlog.shape}")

# sanity check: predicted spatial dims must match mean_pr
assert y_pred_normlog.shape[1:3] == (H, W), (
    f"Spatial dims mismatch: predicted {y_pred_normlog.shape[1:3]} vs mean_pr {(H,W)}"
)

# -----------------------------------------------------------------------------
# 4. Compute test set masked, weighted MAE (normalized log1p)
# -----------------------------------------------------------------------------
y_true_test = tf.convert_to_tensor(y_all[n_tr:], dtype=tf.float32)
y_pred_test = tf.convert_to_tensor(y_pred_normlog[n_tr:], dtype=tf.float32)
test_loss = loss_fn(y_true_test, y_pred_test)
print(f"¶ Test MAE (norm log1p, land only) = {test_loss.numpy():.6f}")

# -----------------------------------------------------------------------------
# 5. Denormalize & invert log1p ’ mm/day
# -----------------------------------------------------------------------------
# broadcast mean/std to (1,H,W,1)
mean_exp = mean_pr[None,:,:,None]
std_exp  = std_pr[None,:,:,None]

# undo normalization
y_pred_logp = y_pred_normlog * std_exp + mean_exp
# undo log1p ’ mm/day
y_pred_mm = np.clip(np.expm1(y_pred_logp), 0, None)
# drop channel axis
y_pred_mm = y_pred_mm[...,0]
print(f"  y_pred_mm.shape = {y_pred_mm.shape}")

# split back train/test
pred_train_mm = y_pred_mm[:n_tr]
pred_test_mm  = y_pred_mm[n_tr:]
print(f"  pred_train_mm.shape = {pred_train_mm.shape}")
print(f"  pred_test_mm.shape  = {pred_test_mm.shape}")

# -----------------------------------------------------------------------------
# 5b. Apply land mask (set ocean to NaN)
# -----------------------------------------------------------------------------
print("  Applying land mask to set ocean areas to NaN...")
ocean_mask = (land_mask == 0)
pred_train_mm[:, ocean_mask] = np.nan
pred_test_mm[:, ocean_mask] = np.nan

# -----------------------------------------------------------------------------
# 6. Save raw numpy and write CF compliant NetCDF
# -----------------------------------------------------------------------------
os.makedirs('output', exist_ok=True)
np.save('output/predicted_train_mm.npy', pred_train_mm)
np.save('output/predicted_test_mm.npy',  pred_test_mm)
print("¶ Saved output/predicted_*_mm.npy")

times_tr = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/high_res_train_time_coords.npy', allow_pickle=True)
times_te = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/high_res_test_time_coords.npy',  allow_pickle=True)
lat      = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/high_res_lat.npy',               allow_pickle=True)
lon      = np.load('/scratch/user/u.hn319322/ondemand/Downscaling/plan1_ablation/mydata/high_res_lon.npy',               allow_pickle=True)

train_ds = xr.Dataset(
    {'precip_mm': (('time','lat','lon'), pred_train_mm)},
    coords={'time': times_tr, 'lat': lat, 'lon': lon},
    attrs={'description':'Prism downscaled precip (train)','units':'mm/day'}
)
test_ds = xr.Dataset(
    {'precip_mm': (('time','lat','lon'), pred_test_mm)},
    coords={'time': times_te, 'lat': lat, 'lon': lon},
    attrs={'description':'Prism downscaled precip (test)', 'units':'mm/day'}
)
for ds in (train_ds, test_ds):
    ds.precip_mm.attrs['long_name'] = 'precipitation rate'

# New, corrected block
try:
    # If you really want to try netcdf4, keep it **without compression** and **no _FillValue**
    train_ds.to_netcdf('output/train_precip_mm.nc', engine='netcdf4',
                       encoding={'precip_mm': {'dtype': 'float32'}})
    test_ds .to_netcdf('output/test_precip_mm.nc',  engine='netcdf4',
                       encoding={'precip_mm': {'dtype': 'float32'}})
    print("¶ Wrote NetCDF with netcdf4 backend (no compression)")
except Exception as e:
    print("netcdf4 failed; falling back to SciPy:", e)
    train_ds.to_netcdf('output/train_precip_mm.nc', engine='scipy', encoding={'precip_mm': {'dtype': 'float32'}})
    test_ds .to_netcdf('output/test_precip_mm.nc',  engine='scipy', encoding={'precip_mm': {'dtype': 'float32'}})
    print("¶ Wrote NetCDF via SciPy backend")


# -----------------------------------------------------------------------------
# 7. Quick obs vs. pred plot
# -----------------------------------------------------------------------------
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
obs0_mm = np.clip(np.expm1(y_te[0] if y_te.ndim==3 else y_te[0,...,0]), 0, None)
ax1.imshow(obs0_mm, origin='upper')
ax1.set_title('Obs (mm/day)')
im2 = ax2.imshow(pred_test_mm[0], origin='upper')
ax2.set_title('Pred (mm/day)')
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.savefig('output/precip_compare_mm.png', dpi=300)
print("¶ Saved comparison: output/precip_compare_mm.png")

print("  Prism postprocessing COMPLETE.")
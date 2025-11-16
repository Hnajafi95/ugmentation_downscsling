import numpy as np
import os
import xarray as xr
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cftime

def safe_normalize(data, mean, std, epsilon=1e-8):
    """Normalizes predictor data, clipping extreme values."""
    safe_std = np.copy(std)
    safe_std[safe_std < epsilon] = epsilon
    normalized = (data - mean) / safe_std
    return np.clip(normalized, -10.0, 10.0)

def normalize_no_clip(data, mean, std, epsilon=1e-8):
    """Normalizes target data without clipping, preserving the tail."""
    safe_std = np.maximum(std, epsilon)
    return (data - mean) / safe_std

def check_for_nans_infs_zeros(data, data_name):
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    zero_count = np.size(data) - np.count_nonzero(data)
    print(f"{data_name} - NaNs: {nan_count}, Infs: {inf_count}, Zeros: {zero_count}")

def fix_orientation(data_array, lat, lon, data_name):
    data_out = np.copy(data_array)
    lat_out = np.copy(lat)
    lon_out = np.copy(lon)

    if lon_out.min() < 0:
        print(f"Fixing orientation for '{data_name}': Converting longitudes to 0-360 range.")
        lon_out = np.where(lon_out < 0, lon_out + 360, lon_out)
        
    if lat_out[0] > lat_out[-1]:
        print(f"Fixing orientation for '{data_name}': Flipping latitude axis to ascending order.")
        data_out = np.flip(data_out, axis=1)
        lat_out = np.flip(lat_out)
        
    if lon_out[0] > lon_out[-1]:
        print(f"Fixing orientation for '{data_name}': Flipping longitude axis to ascending order.")
        data_out = np.flip(data_out, axis=2)
        lon_out = np.flip(lon_out)
        
    return data_out, lat_out, lon_out

def load_nc_variable(file, var_candidates, start_date, end_date):
    ds = xr.open_dataset(file)
    available_vars = list(ds.data_vars)
    var_name = None
    for candidate in var_candidates:
        if candidate in available_vars:
            var_name = candidate
            break
    if var_name is None:
        for candidate in var_candidates:
            for v in available_vars:
                if candidate in v:
                    var_name = v
                    break
            if var_name is not None:
                break
    if var_name is None:
        raise ValueError(f"Could not find any variable among {var_candidates} in file {file}")
    
    print(f"Loading variable '{var_name}' from file {os.path.basename(file)}")
    data = ds[var_name].sel(time=slice(start_date, end_date))
    
    if np.isnan(data.values).any():
        print(f"Warning: {var_name} contains NaNs; replacing with zeros.")
        data = data.fillna(0.0)
    if np.isinf(data.values).any():
        print(f"Warning: {var_name} contains Infs; replacing with zeros.")
        data = xr.where(np.isinf(data), 0.0, data)
    
    if isinstance(data.time.values[0], cftime.datetime):
        time_coords = pd.to_datetime([t.isoformat() for t in data.time.values])
    else:
        time_coords = pd.to_datetime(data.time.values)
        
    lat = data['lat'].values
    lon = data['lon'].values
    data_array = data.values
    data_array, lat, lon = fix_orientation(data_array, lat, lon, var_name)
    ds.close()
    return data_array, time_coords, lat, lon, var_name

def process_low_res(directory, start_date, end_date, variables_to_find):
    files = glob.glob(os.path.join(directory, "*.nc"))
    if not files:
        raise FileNotFoundError(f"No netCDF files found in {directory}")
    
    low_res_data_vars = []
    time_coords_list = []
    lat_list = []
    lon_list = []
    var_names = []
    
    for var in variables_to_find:
        matching_file = None
        for f in files:
            if os.path.basename(f).startswith(f"{var}_day_"):
                matching_file = f
                break
        if matching_file is None:
            print(f"Warning: No file found for variable {var} in {directory}; skipping.")
            continue
        
        data_array, time_coords, lat, lon, loaded_var = load_nc_variable(matching_file, [var], start_date, end_date)
        low_res_data_vars.append(data_array)
        time_coords_list.append(time_coords)
        lat_list.append(lat)
        lon_list.append(lon)
        var_names.append(loaded_var)
    
    if not low_res_data_vars:
        raise ValueError("No variables were loaded from the low-res data.")
    
    common_times = time_coords_list[0]
    for t in time_coords_list[1:]:
        common_times = np.intersect1d(common_times, t)
    common_times = pd.to_datetime(common_times)
    print(f"Common time steps in low-res data: {len(common_times)}")
    
    low_res_data_vars_aligned = []
    for data_array, t in zip(low_res_data_vars, time_coords_list):
        original_days = np.array([np.datetime64(x, 'D') for x in t])
        common_days = np.array([np.datetime64(x, 'D') for x in common_times])
        mask = np.isin(original_days, common_days)
        low_res_data_vars_aligned.append(data_array[mask, ...])
    
    time_shapes = [d.shape[0] for d in low_res_data_vars_aligned]
    if len(set(time_shapes)) != 1:
        raise ValueError("Time dimensions do not match across low-res variables after alignment.")
    
    ref_lat = lat_list[0]
    ref_lon = lon_list[0]
    for lat in lat_list[1:]:
        if not np.allclose(ref_lat, lat):
            raise ValueError("Latitude arrays do not match across low-res variables.")
    for lon in lon_list[1:]:
        if not np.allclose(ref_lon, lon):
            raise ValueError("Longitude arrays do not match across low-res variables.")
    
    low_res_data = np.stack(low_res_data_vars_aligned, axis=-1)
    return low_res_data, common_times, ref_lat, ref_lon, var_names

def process_high_res(high_res_file, start_date, end_date, var_candidates):
    if not os.path.exists(high_res_file):
        raise FileNotFoundError(f"High-res file not found: {high_res_file}")

    ds = xr.open_dataset(high_res_file)
    available = list(ds.data_vars)
    var_name = 'Precip' if 'Precip' in available else available[0]
    raw = ds[var_name].sel(time=slice(start_date, end_date))

    original_lat = raw['lat'].values
    original_lon = raw['lon'].values
    raw_values_with_nans = raw.values

    land_mask_unoriented = ~np.isnan(raw_values_with_nans)
    land_mask, _, _ = fix_orientation(land_mask_unoriented, original_lat, original_lon, "land_mask")

    raw_values_oriented, _, _ = fix_orientation(raw_values_with_nans, original_lat, original_lon, "raw_values_for_check")
    reference_mask_from_oriented_data = ~np.isnan(raw_values_oriented)

    if not np.array_equal(land_mask, reference_mask_from_oriented_data):
        mismatch = np.sum(land_mask != reference_mask_from_oriented_data)
        print(f"❌ SANITY CHECK FAILED: Land mask alignment is incorrect! Mismatched pixels: {mismatch}")
    else:
        print("✅ Sanity check passed: The re-oriented land mask is correctly aligned with the data.")

    filled = raw.fillna(0.0)
    data = filled.values
    data = np.where(np.isfinite(data), data, 0.0)

    data, lat, lon = fix_orientation(data, original_lat, original_lon, var_name)
    
    if isinstance(raw.time.values[0], cftime.datetime):
        time_coords = pd.to_datetime([t.isoformat() for t in raw.time.values])
    else:
        time_coords = pd.to_datetime(raw.time.values)

    ds.close()
    return data, time_coords, lat, lon, var_name, land_mask

def match_time_coordinates(low_res_times, high_res_times):
    low_res_days = np.array([np.datetime64(t, 'D') for t in low_res_times])
    high_res_days = np.array([np.datetime64(t, 'D') for t in high_res_times])
    common_times = np.intersect1d(low_res_days, high_res_days)
    if len(common_times) == 0:
        raise ValueError("No matching time coordinates between low-res and high-res data.")
    print(f"Found {len(common_times)} common time steps between low-res and high-res data.")
    return pd.to_datetime(common_times)

def subset_data_by_times(data, original_times, common_times):
    original_days = np.array([np.datetime64(t, 'D') for t in original_times])
    common_days = np.array([np.datetime64(t, 'D') for t in common_times])
    mask = np.isin(original_days, common_days)
    return data[mask, ...]

def apply_precipitation_log_transform(data, var_names):
    pr_index = None
    for i, var in enumerate(var_names):
        if var.lower().startswith('pr'):
            pr_index = i
            break
    if pr_index is not None:
        print(f"Applying log1p transformation to precipitation at index {pr_index} ({var_names[pr_index]}).")
        data[..., pr_index] = np.where(data[..., pr_index] < 0, 0, np.log1p(data[..., pr_index]))
    else:
        print("No precipitation variable found for log transformation.")
    return data

def visualize_data(high_res_data, low_res_data, high_res_lat, high_res_lon, low_res_lat, low_res_lon, day_index=50, output_path='output/preprocessed_data_comparison.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    high_sample = high_res_data[day_index]
    if high_sample.ndim == 3:
        high_sample = high_sample[..., 0]
    im1 = ax1.imshow(high_sample, cmap='viridis', 
                     extent=[high_res_lon.min(), high_res_lon.max(), high_res_lat.min(), high_res_lat.max()],
                     aspect='auto', origin='upper')
    ax1.set_title('High-Resolution Sample')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(im1, ax=ax1, label='Value')
    
    low_sample = low_res_data[day_index]
    if low_sample.ndim == 3:
        low_sample = low_sample[..., 0]
    im2 = ax2.imshow(low_sample, cmap='viridis',
                     extent=[low_res_lon.min(), low_res_lon.max(), low_res_lat.min(), low_res_lat.max()],
                     aspect='auto', origin='upper')
    ax2.set_title('Low-Resolution Sample')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(im2, ax=ax2, label='Value')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Define time periods.
    train_start = '1981-01-01'
    train_end = '2014-12-31'
    test_start = '2015-01-01'
    test_end = '2019-12-31'
    # Define directories and high-res file.
    train_low_res_dir = './lowres-files-train'
    test_low_res_dir = './lowres-files-test'
    high_res_dir = './highres-files'
    high_res_file = os.path.join(high_res_dir, 'PRISM_daily_FL_1981-2019_remapped.nc')
    
    variables_to_find = ['tas', 'pr', 'huss', 'sfcWind', 'tasmax', 'tasmin']
    
    # Process low-res training data.
    print("Processing low-res training data...")
    train_low_res, train_low_res_times, low_res_lat, low_res_lon, low_res_var_names = process_low_res(
        train_low_res_dir, train_start, train_end, variables_to_find)
    
    # Process low-res testing data.
    print("Processing low-res testing data...")
    test_low_res, test_low_res_times, low_res_lat_test, low_res_lon_test, _ = process_low_res(
        test_low_res_dir, test_start, test_end, variables_to_find)
    
    if not (np.allclose(low_res_lat, low_res_lat_test) and np.allclose(low_res_lon, low_res_lon_test)):
        raise ValueError("Low-res coordinates differ between training and testing data.")
    
    # Process high-res training data.
    print("Processing high-res training data...")
    train_high_res, train_high_res_times, high_res_lat, high_res_lon, high_res_var_name, train_mask = process_high_res(
        high_res_file, train_start, train_end, ['Precip'])
    
    # Process high-res testing data.
    print("Processing high-res testing data...")
    test_high_res, test_high_res_times, high_res_lat_test, high_res_lon_test, _, _  = process_high_res(
        high_res_file, test_start, test_end, ['Precip'])
    
    if not (np.allclose(high_res_lat, high_res_lat_test) and np.allclose(high_res_lon, high_res_lon_test)):
        raise ValueError("High-res coordinates differ between training and testing data.")
    
    # Match time coordinates.
    print("Matching training time coordinates between low-res and high-res...")
    common_train_times = match_time_coordinates(train_low_res_times, train_high_res_times)
    train_low_res = subset_data_by_times(train_low_res, train_low_res_times, common_train_times)
    train_high_res = subset_data_by_times(train_high_res, train_high_res_times, common_train_times)
    
    print("Matching testing time coordinates between low-res and high-res...")
    common_test_times = match_time_coordinates(test_low_res_times, test_high_res_times)
    test_low_res = subset_data_by_times(test_low_res, test_low_res_times, common_test_times)
    test_high_res = subset_data_by_times(test_high_res, test_high_res_times, common_test_times)
    
    # Apply log transformation to precipitation.
    train_low_res = apply_precipitation_log_transform(train_low_res, low_res_var_names)
    test_low_res = apply_precipitation_log_transform(test_low_res, low_res_var_names)

    train_high_res = np.where(train_high_res < 0, 0, np.log1p(train_high_res))
    test_high_res  = np.where(test_high_res  < 0, 0, np.log1p(test_high_res))

    # Compute normalization parameters.
    print("Computing normalization parameters for low-res data...")
    low_res_mean = np.mean(train_low_res, axis=0)
    low_res_std = np.std(train_low_res, axis=0, ddof=1)
    low_res_std = np.maximum(low_res_std, 1e-8)
    
    print("Computing normalization parameters for high-res data (land only)...")
    land_mask_2d = train_mask[0].astype(bool)

    high_res_mean = np.mean(train_high_res, axis=0, where=land_mask_2d)
    high_res_std = np.std(train_high_res, axis=0, where=land_mask_2d, ddof=1)

    high_res_mean = np.nan_to_num(high_res_mean)
    high_res_std = np.nan_to_num(high_res_std)
    high_res_std = np.maximum(high_res_std, 1e-8)
    
    # Normalize data.
    print("Normalizing low-res and high-res data...")
    train_low_res_norm = safe_normalize(train_low_res, low_res_mean, low_res_std)
    test_low_res_norm = safe_normalize(test_low_res, low_res_mean, low_res_std)
    train_high_res_norm = normalize_no_clip(train_high_res, high_res_mean, high_res_std)
    test_high_res_norm  = normalize_no_clip(test_high_res,  high_res_mean, high_res_std)

    # Sanity check for tail compression on the training set
    print("Performing tail check on normalized high-resolution training data...")
    T, H, W = train_high_res_norm.shape[:3]
    land_flat = land_mask_2d.reshape(-1)
    vals = train_high_res_norm.reshape(T, -1)[:, land_flat].ravel()
    z99 = np.percentile(vals, 99)
    z999 = np.percentile(vals, 99.9)
    print(f"Tail Check: HR z[99%]={z99:.2f}, z[99.9%]={z999:.2f}")

    check_for_nans_infs_zeros(train_low_res_norm, "Train Low-Res Normalized")
    check_for_nans_infs_zeros(train_high_res_norm, "Train High-Res Normalized")
    
    # Create index arrays and date strings.
    train_indices = np.arange(len(common_train_times))
    test_indices = np.arange(len(common_test_times))
    train_dates = np.array([pd.to_datetime(t).strftime('%Y-%m-%d') for t in common_train_times])
    test_dates = np.array([pd.to_datetime(t).strftime('%Y-%m-%d') for t in common_test_times])
    
    # Save files.
    save_dir = "mydata_corrected"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "predictors_train_mean_std_separate.npy"), train_low_res_norm)
    np.save(os.path.join(save_dir, "predictors_test_mean_std_separate.npy"), test_low_res_norm)
    np.save(os.path.join(save_dir, "obs_train_mean_std_single.npy"), train_high_res_norm)
    np.save(os.path.join(save_dir, "obs_test_mean_std_single.npy"), test_high_res_norm)
    np.save(os.path.join(save_dir, "ERA5_mean_train.npy"), high_res_mean)
    np.save(os.path.join(save_dir, "ERA5_std_train.npy"), high_res_std)
    np.save(os.path.join(save_dir, "train_time_coords.npy"), np.array(common_train_times, dtype='datetime64'))
    np.save(os.path.join(save_dir, "test_time_coords.npy"), np.array(common_test_times, dtype='datetime64'))
    np.save(os.path.join(save_dir, "land_mask.npy"), land_mask_2d)

    np.save(os.path.join(save_dir, "high_res_train_time_coords.npy"), np.array(common_train_times, dtype='datetime64'))
    np.save(os.path.join(save_dir, "high_res_test_time_coords.npy"), np.array(common_test_times, dtype='datetime64'))
    np.save(os.path.join(save_dir, "low_res_lat.npy"), low_res_lat)
    np.save(os.path.join(save_dir, "low_res_lon.npy"), low_res_lon)
    np.save(os.path.join(save_dir, "high_res_lat.npy"), high_res_lat)
    np.save(os.path.join(save_dir, "high_res_lon.npy"), high_res_lon)
    np.save(os.path.join(save_dir, "variables.npy"), np.array(low_res_var_names, dtype='object'))
    np.save(os.path.join(save_dir, "train_indices.npy"), train_indices)
    np.save(os.path.join(save_dir, "test_indices.npy"), test_indices)
    np.save(os.path.join(save_dir, "train_dates.npy"), train_dates)
    np.save(os.path.join(save_dir, "test_dates.npy"), test_dates)
    
    print(f"All processed numpy files saved successfully to '{save_dir}'.")
    
    # Visualization to verify alignment.
    print("Generating visualization for training data...")
    visualize_data(train_high_res_norm, train_low_res_norm,
                   high_res_lat, high_res_lon,
                   low_res_lat, low_res_lon,
                   day_index=0,
                   output_path=os.path.join("output", "corrected_data_comparison.png"))
    
    print("Preprocessing completed successfully.")

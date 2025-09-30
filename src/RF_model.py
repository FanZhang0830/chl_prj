# This script builds a random forest model to predict chlorophyll-a concentration and saves the trained model in trained_models folder.
# %%
# import necessary libraries
import osgeo
import os
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import rasterio
from rasterio.transform import from_origin

# %%
# find the root directory of the git repository
from pathlib import Path


def find_repo_root(path):
    """
    Finds the root directory of a Git repository by traversing parent directories.

    Args:
        path (str or Path): The starting path to search from.

    Returns:
        Path or None: The Path object representing the repository root,
                      or None if no .git directory is found in parent directories.
    """
    current_path = Path(
        path
    ).resolve()  # Resolve to get absolute path and handle symlinks
    for parent in [current_path] + list(current_path.parents):
        git_dir = parent / ".git"
        if git_dir.is_dir():
            return parent
    return None


# Example Usage:
# Find the repository root where the current script is located
repo_root = find_repo_root(__file__)
if repo_root:
    print(f"Repository root found at: {repo_root}")
else:
    print("Not within a Git repository.")


# %%
# set the working directory to the root of the repository
os.chdir(repo_root)
chl_df = pd.read_csv("data\\ch_train_test_0706.csv")

# Example usage
tiff_file = "data\\raw_masked_image_0706.tif"
target_crs = "EPSG:4326"

# %%
# using rioxarray to process the data
# define xarray
ds = rioxarray.open_rasterio(tiff_file)

s2_image = ds
B3_B2 = s2_image.sel(band=3) / s2_image.sel(band=2)
B4_B3 = s2_image.sel(band=4) / s2_image.sel(band=3)
B5_B4 = s2_image.sel(band=5) / s2_image.sel(band=4)
ndvi = (s2_image.sel(band=8) - s2_image.sel(band=4)) / (
    s2_image.sel(band=8) + s2_image.sel(band=4)
)

# Add the band ratios to the DataArray
# Create a new DataArray with an additional 'band' dimension for the ratios
new_bands = xr.concat([B3_B2, B4_B3, B5_B4, ndvi], dim="band")
new_bands = new_bands.assign_coords(band=[12, 13, 14, 15])  # Assign new band numbers

# Combine the original and new bands into one DataArray
s2_image = xr.concat([s2_image, new_bands], dim="band")

# reproject the xarray
reprojected_ds = s2_image.rio.reproject(target_crs)


# import the geometry data of train_test data
tgt_x = xr.DataArray(chl_df["Longitude"], dims="points")
tgt_y = xr.DataArray(chl_df["Latitude"], dims="points")

# extract band values at each points
band_values = reprojected_ds.sel(
    band=[12, 13, 14, 15], x=tgt_x, y=tgt_y, method="nearest"
).values


# extract ch_in_situ values at each points
ch_in_situ = chl_df["ch_in_situ"]

# build the random forest model to fit different band values with ch_in_situ

X = pd.DataFrame(band_values.T, columns=("B3_B2", "B4_B3", "B5_B4", "ndvi"))
y = ch_in_situ

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=83
)

rf_regressor = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=83)

clf = GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.1, max_depth=5, random_state=53
).fit(X_train, y_train)
print("check clf accuarcy: ", clf.score(X_test, y_test))

# Fitting the model on the training data
rf_regressor.fit(X_train, y_train)

# Predicting on the test data
y_pred = rf_regressor.predict(X_test)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# Calculate R-squared (coefficient of determination)
r_squared = r2_score(y_test, y_pred)
print("R-squared (R^2) value:", r_squared)

# %%
#### Save the model for future use ######
import joblib

joblib.dump(rf_regressor, "trained_models\\rf_model.pkl")
rf2 = joblib.load("trained_models\\rf_model.pkl")
rf2.predict(X_test[0:5])
######################################

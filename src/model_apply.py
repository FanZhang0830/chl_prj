### apply the trained and tested model on the remote sensing images

# Convert xarray.DataArray to pandas DataFrame and apply rf_regressor to calcualte the chl concentration results
# re-take Sentinel-2 image
reprojected_ds = ds.rio.reproject(target_crs)
# mask non-value parts
reprojected_ds = reprojected_ds.where(reprojected_ds != 65535, drop=True)

# mask non-water parts
# Calculate normalized difference
normalized_diff = (reprojected_ds.sel(band=2) - reprojected_ds.sel(band=7)) / (
    reprojected_ds.sel(band=2) + reprojected_ds.sel(band=7)
)

# Define threshold for masking
threshold = 0.0  # Adjust this threshold as per your requirement

# Apply mask
reprojected_ds = reprojected_ds.where(normalized_diff > threshold, drop=True)

# check the reprojected ds
reprojected_ds.sel(band=[1], method="nearest").plot()
plt.show()


df = reprojected_ds.to_dataframe(name="value").reset_index()

# Extract coordinates (x, y) to a separate column
df["coords"] = list(zip(df["x"], df["y"]))

# Drop redundant 'x' and 'y' columns
df.drop(columns=["x", "y"], inplace=True)

# Group by 'coords' and aggregate values for each band
df_combined = df.groupby("coords")["value"].apply(list).reset_index()

# Unpack the aggregated values into separate columns for each band
for i in range(1, len(reprojected_ds.coords["band"]) + 1):
    df_combined[f"band{i}"] = df_combined["value"].apply(lambda x: x[i - 1])

# Drop the 'value' column
df_combined.drop(columns=["value"], inplace=True)
df_combined.columns = list(["coords"]) + X.columns.tolist()

# apply the rf_regressor to each band values
# fill nan values with inf values
# df_combined.fillna(1000000)

df_combined.dropna(inplace=True)

see = rf_regressor.predict(df_combined.iloc[:, 1:])

df_combined["chl_model"] = see
# visualize the chlorophyll concentraion map

# Extracting latitude and longitude
df_combined["Latitude"] = df_combined["coords"].apply(lambda x: x[1])
df_combined["Longitude"] = df_combined["coords"].apply(lambda x: x[0])

# Plotting the map
# plt.scatter(df_combined['Longitude'], df_combined['Latitude'], c=df_combined['chl_model'], cmap='seismic')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Map with Color-Coded Values')
# plt.colorbar(label='Value')
# plt.show()

# output the raster as netcdf file
df_toxr = df_combined[["Latitude", "Longitude", "chl_model"]]
ds_output = df_toxr.set_index(["Latitude", "Longitude"]).to_xarray()

ds_output["chl_model"].plot(x="Longitude", y="Latitude", cmap="seismic")
plt.show()

# ds_output.to_netcdf('output_raster.nc')

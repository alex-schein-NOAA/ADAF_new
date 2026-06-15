import argparse
import os
import sys
import datetime as dt

# Data Science Core
import numpy as np
import pandas as pd
import xarray as xr

# Geospatial & Analysis
# import xesmf
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from scipy.spatial import KDTree
# from metpy.calc import specific_humidity_from_dewpoint
# from metpy.units import units

# Plotting
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# from matplotlib import colors
# from matplotlib.markers import MarkerStyle

# Custom Packages
# from nnja_ai import DataCatalog
from funcs_modified import *

###############################

# (2026-05-07) barebones for now but will get more added in when satellite issues are solved

parser = argparse.ArgumentParser(description="")
parser.add_argument("--starting_analysis_time", type=str) #Must be formatted as "YYYY-MM-DD_HH"
parser.add_argument("--ending_analysis_time", type=str) #Must be formatted as "YYYY-MM-DD_HH"
parser.add_argument("--save_directory", type=str, default=None)
parser.add_argument("--obs_source", type=str, default="metar", choices=["metar", "ioda"],
                    help="Station obs source: 'metar' (nnja-ai METAR, default) or 'ioda' (our prepBUFR->IODA mesonet).")
parser.add_argument("--ioda_root", type=str, default=IODA_ROOT,
                    help="Root dir holding per-cycle IODA files (<ioda_root>/run_<YYYYMMDDHH>/ioda_msonet.nc). Only used when --obs_source=ioda.")


args = parser.parse_args()
starting_analysis_time = dt.datetime.strptime(args.starting_analysis_time, "%Y-%m-%d_%H")
ending_analysis_time = dt.datetime.strptime(args.ending_analysis_time, "%Y-%m-%d_%H")
save_directory = args.save_directory
obs_source = args.obs_source
ioda_root = args.ioda_root

if save_directory is None:
   sys.exit("ERROR: --save_directory must be specifed!")
   

analysis_times_list = pd.date_range(start=starting_analysis_time, end=ending_analysis_time, freq='h').to_pydatetime().tolist()


### Vars shared between HRRR/RTMA and sta
adaf_var_list = ['sp', 't', 'q', 'u10', 'v10']

adaf_grid_spacing = 0.05
adaf_lats = (np.arange(24.7, 50.25+adaf_grid_spacing, adaf_grid_spacing)).round(4)
adaf_lons = (np.arange(232, 295.95+adaf_grid_spacing, adaf_grid_spacing)).round(4)
ds_adaf_lats_lons = xr.Dataset(coords={"lat": adaf_lats, "lon": adaf_lons})
df_adaf_lats_lons = (pd.MultiIndex.from_product([adaf_lats, adaf_lons], names=["lat","lon"])).to_frame(index=False)

stats_filepath=f"/scratch3/BMC/wrfruc/aschein/ADAF/data_preparation_new/stats.csv"
stats = pd.read_csv(stats_filepath, index_col=0)

### HRRR/RTMA specific vars, static
hrrr_forecast_leadtime = 1

rtma_variables = [f"rtma_{x}" for x in adaf_var_list] 
hrrr_variables = [f"hrrr_{x}" for x in adaf_var_list] 

#Static terrain field
topo = get_grid_alt_and_regrid(f"/scratch3/BMC/wrfruc/aschein/Train_Test_Files/terrain_CONUS_URMA_2p5km.grib2", ds_adaf_lats_lons)
topo_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(topo, 'z', stats_filepath)

### Station specific vars, static
obs_time_window = 3 #hours
time_period = 14 #days
catalog_str = 'conv-adpsfc-NC000007'
list_of_metar_vars = ["LAT","LON", "OBS_TIMESTAMP", "MTRTMP.TMDB", "MTRTMP.TMDP", "MTRPRS.ALSE", "MTRWND.WSPD", "MTRWND.WDIR"]

#Initialize outside the main loop (METAR only; the IODA path reads per-cycle files inside the loop)
if obs_source == "metar":
    df_master = get_nnja_metar_dataframe(analysis_times_list[0], list_of_metar_vars, time_period=time_period, catalog_str=catalog_str)

written_count = 0
already_exists_count = 0
skipped_missing_obs_count = 0
skipped_empty_sta_count = 0

for t, analysis_time in enumerate(analysis_times_list):
    output_filename = f"{analysis_time.strftime("%Y-%m-%d_%H")}.nc"
    if os.path.exists(f"{save_directory}/{output_filename}"):
        print(f"{output_filename} already exists in {save_directory}")
        already_exists_count += 1
    else:
        hrrr_init_time = analysis_time - dt.timedelta(hours=hrrr_forecast_leadtime) #need to call the proper f01 HRRR file
        
        # Dynamic file directories 
        hrrr_directory = f"/scratch5/BMC/ai-datadepot/data/models/hrrr/conus/grib2/{hrrr_init_time.strftime("%Y%m%d")}"
        rtma_directory=f"/scratch5/BMC/ai-datadepot/data/models/rtma/2p5km/grib2/{analysis_time.strftime("%Y%m%d")}" #2026-05-29 updated to the main depot
    
        ### Generate all RTMA, HRRR fields
        hrrr_data = []
        rtma_data = []
        
        for i, adaf_var in enumerate(adaf_var_list):
        
            hrrr_filename = f"hrrr.t{str(hrrr_init_time.hour).zfill(2)}z.wrfnatf01.grib2"
            rtma_filename = f"rtma2p5.t{str(analysis_time.hour).zfill(2)}z.2dvaranl_ndfd.grb2_wexp"

            xr_hrrr_regridded, xr_rtma_regridded = fetch_and_regrid_hrrr_rtma(adaf_var, 
                                                                              ds_adaf_lats_lons,
                                                                              hrrr_filepath=f"{hrrr_directory}/{hrrr_filename}", 
                                                                              rtma_filepath=f"{rtma_directory}/{rtma_filename}")      
        
            xr_hrrr_fixed = fix_dataset_scaling_shifting(xr_hrrr_regridded, adaf_var)
            xr_rtma_fixed = fix_dataset_scaling_shifting(xr_rtma_regridded, adaf_var)
        
            xr_rtma_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_rtma_fixed, 
                                                                                rtma_variables[i], 
                                                                                stats_filepath)
        
            xr_hrrr_normed = min_max_norm_ignore_extreme_fill_nan_onevar_onetime(xr_hrrr_fixed, 
                                                                                hrrr_variables[i], 
                                                                                stats_filepath)
        
            hrrr_data.append(xr_hrrr_normed.data)
            rtma_data.append(xr_rtma_normed.data)
    
        ds_hrrr_rtma = xr.Dataset(
                {
                # %% rtma, normalized
                f"{rtma_variables[0]}": (("lat", "lon"), rtma_data[0]),
                f"{rtma_variables[1]}": (("lat", "lon"), rtma_data[1]),
                f"{rtma_variables[2]}": (("lat", "lon"), rtma_data[2]),
                f"{rtma_variables[3]}": (("lat", "lon"), rtma_data[3]),
                f"{rtma_variables[4]}": (("lat", "lon"), rtma_data[4]),
                
                # %% HRRR, now normalized (as of 2026/4/27)
                f"{hrrr_variables[0]}": (("lat", "lon"), hrrr_data[0]),
                f"{hrrr_variables[1]}": (("lat", "lon"), hrrr_data[1]),
                f"{hrrr_variables[2]}": (("lat", "lon"), hrrr_data[2]),
                f"{hrrr_variables[3]}": (("lat", "lon"), hrrr_data[3]),
                f"{hrrr_variables[4]}": (("lat", "lon"), hrrr_data[4]),
    
                #topography, normalized
                f"z" : (("lat", "lon"), topo_normed.data),
                },
            coords={
                "valid_time": analysis_time,
                "lat": adaf_lats,
                "lon": adaf_lons 
                    },
            )
    
        ### Station obs
        if obs_source == "ioda":
            # Each IODA file is one cycle (hour); a sample needs the cycles covering
            # required hours [t-(window-1) .. t]. Skip if any of those files is absent.
            required_cycles = [analysis_time - dt.timedelta(hours=h) for h in range(obs_time_window - 1, -1, -1)]
            missing_cycles = [c for c in required_cycles if not os.path.exists(ioda_cycle_path(c, ioda_root))]
            if len(missing_cycles) > 0:
                missing_str = ", ".join([c.strftime("%Y-%m-%d %H") for c in missing_cycles])
                print(f"Skipping {output_filename}: missing required IODA cycle file(s) -> {missing_str}")
                skipped_missing_obs_count += 1
                continue
            df_master = get_ioda_mesonet_dataframe(analysis_time, obs_time_window=obs_time_window, ioda_root=ioda_root)
            df = assemble_complete_ioda_df(df_master, analysis_time, obs_time_window, df_adaf_lats_lons)
        else:
            df_master = check_nnja_metar_dataframe(df_master, analysis_time, list_of_metar_vars)
            missing_obs_hours = get_missing_obs_hours(df_master, analysis_time, obs_time_window=obs_time_window)
            if len(missing_obs_hours) > 0:
                missing_str = ", ".join([x.strftime("%Y-%m-%d %H:%M") for x in missing_obs_hours])
                print(f"Skipping {output_filename}: missing required obs hour(s) for station window -> {missing_str}")
                skipped_missing_obs_count += 1
                continue

            df = assemble_complete_metar_df(df_master, analysis_time, obs_time_window, df_adaf_lats_lons)

        if df.empty:
            print(f"Skipping {output_filename}: no valid station observations remained after filtering/QC.")
            skipped_empty_sta_count += 1
            continue

        df = min_max_norm_ignore_extreme_fill_nan_sta_df(df, stats_path=stats_filepath)
        ds_sta_obs = create_obs_xarray(df, df_adaf_lats_lons, analysis_time)

        ### Merge and compress and save
        ds = xr.merge([ds_hrrr_rtma, ds_sta_obs], compat="no_conflicts")

        comp_settings = {"zlib": True, "complevel": 1}
        encoding = {var: comp_settings for var in ds.data_vars}

        ds.to_netcdf(f"{save_directory}/{output_filename}", encoding=encoding)
        print(f"{output_filename} saved to {save_directory}")
        written_count += 1

print(
    "Run summary: "
    f"requested={len(analysis_times_list)}, "
    f"written={written_count}, "
    f"already_exists={already_exists_count}, "
    f"skipped_missing_obs_window={skipped_missing_obs_count}, "
    f"skipped_empty_station_obs={skipped_empty_sta_count}"
)

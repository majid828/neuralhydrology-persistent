import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo import camelsus
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class HourlyCamelsUS(camelsus.CamelsUS):
    """Data set class providing hourly data for CAMELS US basins.

    MODIFICATION (15-min support):
    - If the run requests 15-minute frequency (cfg.use_frequencies contains '15min'),
      then we:
        1) load HOURLY forcings (as before),
        2) upsample forcings to 15-min by repeating each hourly value 4 times (ffill),
        3) split APCP_surface into 4 equal 15-min portions (divide by 4),
        4) load TRUE 15-min QObs as the target and join to the 15-min forcing dataframe.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: list = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        self._netcdf_datasets = {}  # if available, we remember the dataset to load faster
        self._warn_slow_loading = True
        super(HourlyCamelsUS, self).__init__(cfg=cfg,
                                             is_train=is_train,
                                             period=period,
                                             basin=basin,
                                             additional_features=additional_features,
                                             id_to_int=id_to_int,
                                             scaler=scaler)

    # ---------------------------
    # Helper: detect 15-min mode
    # ---------------------------
    def _is_15min_run(self) -> bool:
        """Return True if the run requests a 15-minute frequency."""
        freqs = getattr(self.cfg, "use_frequencies", None)
        if not freqs:
            return False
        # Normalize to strings (cfg may store offsets, strings, etc.)
        freqs_str = [str(f).lower() for f in freqs]
        return any(("15min" in f) or (f == "15t") for f in freqs_str)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data.

        HOURLY mode (unchanged behavior):
          - Load hourly forcings + hourly discharge (qobs) from hourly CAMELS-US.

        15-min mode (new behavior):
          - Load HOURLY forcings (do NOT rely on hourly qobs),
          - Upsample forcings to 15-min by repeating values (ffill),
          - Divide APCP_surface by 4 (hourly total -> four 15-min totals),
          - Join TRUE 15-min QObs target from /mnt/disk1/CAMELS_US/15min.
        """
        run_15min = self._is_15min_run()

        # -------------------------
        # 1) Load FORCINGS (hourly)
        # -------------------------
        dfs = []
        if not any(f.endswith('_hourly') for f in self.cfg.forcings):
            raise ValueError('Forcings include no hourly forcings set.')

        for forcing in self.cfg.forcings:
            if forcing.endswith('_hourly'):
                # In 15-min mode, we DO NOT need hourly discharge, because we will load 15-min QObs.
                df = self.load_hourly_data(basin, forcing, load_discharge=not run_15min)
            else:
                # Load daily CAMELS forcings and upsample to hourly (repeat each daily value across hours)
                df, _ = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, forcing)
                df = df.resample('1h').ffill()

            # If multiple forcing products are used, rename columns to keep them unique.
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={
                    col: f"{col}_{forcing}" for col in df.columns if 'qobs' not in col.lower()
                })
            dfs.append(df)

        # Combine all forcing dfs into one dataframe (columns = all features).
        df = pd.concat(dfs, axis=1)

        # --------------------------------------------
        # 2) Optionally add daily discharge QObs(mm/d)
        # --------------------------------------------
        # This block is from original code; keep it.
        # It is unrelated to the 15-min QObs(mm/h) target.
        all_features = self.cfg.target_variables
        if isinstance(self.cfg.dynamic_inputs, dict):
            for val in self.cfg.dynamic_inputs.values():
                all_features = all_features + val
        elif isinstance(self.cfg.dynamic_inputs, list):
            all_features = all_features + self.cfg.dynamic_inputs_flattened

        # catch also QObs(mm/d)_shiftX or _copyX features
        if any([x.startswith("QObs(mm/d)") for x in all_features]):
            # add daily discharge from CAMELS, using daymet to get basin area
            _, area = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, "daymet")
            discharge = camelsus.load_camels_us_discharge(self.cfg.data_dir, basin, area)
            discharge = discharge.resample('1h').ffill()  # repeat daily to hourly
            df["QObs(mm/d)"] = discharge

        # only warn for missing netcdf files once for each forcing product
        self._warn_slow_loading = False

        # ------------------------------------------
        # 3) Clean invalid discharge values (hourly)
        # ------------------------------------------
        # In 15-min mode, hourly qobs might be absent (we skipped it), but this is safe.
        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        # -------------------------------------------------
        # 4) Optional: stage (kept as original behavior)
        # -------------------------------------------------
        if 'gauge_height_m' in self.cfg.target_variables:
            df = df.join(load_hourly_us_stage(self.cfg.data_dir, basin))
            df.loc[df['gauge_height_m'] < 0, 'gauge_height_m'] = np.nan

        # ----------------------------------------------------------
        # 5) Optional: synthetic stage (kept as original behavior)
        # ----------------------------------------------------------
        if 'synthetic_qobs_stage_meters' in self.cfg.target_variables:
            attributes = camelsus.load_camels_us_attributes(data_dir=self.cfg.data_dir, basins=[basin])
            with open(self.cfg.rating_curve_file, 'rb') as f:
                rating_curves = pickle.load(f)
            df['synthetic_qobs_stage_meters'] = np.nan
            if basin in rating_curves.keys():
                # Uses hourly qobs in mm/hour -> converts to m3/s using basin area
                discharge_m3s = df['qobs_mm_per_hour'].values / 1000 * attributes.area_gages2[basin] * 1e6 / 60**2
                df['synthetic_qobs_stage_meters'] = rating_curves[basin].discharge_to_stage(discharge_m3s)

        # ==========================================================
        # 6) NEW: 15-min mode conversion (THIS IS YOUR MAIN CHANGE)
        # ==========================================================
        if run_15min:
            # (a) Upsample hourly forcings to 15-min by repeating each hourly value 4 times.
            #     Example: if TMP at 01:00 is X, then 01:00, 01:15, 01:30, 01:45 will all be X.
            df = df.resample("15min").ffill()

            # (b) Split hourly precipitation accumulation into four equal 15-min chunks.
            #     Example: if APCP_surface at 01:00 is 4 mm/hour total, then each 15-min step gets 1 mm.
            if "APCP_surface" in df.columns:
                df["APCP_surface"] = df["APCP_surface"] / 4.0

            # (c) Load TRUE 15-min discharge (QObs) and join to the dataframe.
            #     This ensures your target really is 15-min, not hourly repeated.
            q15 = load_15min_us_discharge(Path("/mnt/disk1/CAMELS_US/15min"), basin)

            # Inner join keeps only timestamps where both forcings and QObs exist.
            df = df.join(q15, how="inner")

            # (d) Replace invalid discharge values with NaNs (same rule as hourly)
            qobs_15_cols = [col for col in df.columns if 'qobs' in col.lower()]
            for col in qobs_15_cols:
                df.loc[df[col] < 0, col] = np.nan

        return df

    def load_hourly_data(self, basin: str, forcings: str, load_discharge: bool = True) -> pd.DataFrame:
        """Load a single set of hourly forcings (and optionally hourly discharge).

        MODIFICATION:
        - Added parameter load_discharge:
            * hourly run: load_discharge=True (original behavior)
            * 15-min run: load_discharge=False (we will load true 15-min QObs later)
        """
        fallback_csv = False
        try:
            if forcings not in self._netcdf_datasets.keys():
                self._netcdf_datasets[forcings] = load_hourly_us_netcdf(self.cfg.data_dir, forcings)
            df = self._netcdf_datasets[forcings].sel(basin=basin).to_dataframe()
        except FileNotFoundError:
            fallback_csv = True
            if self._warn_slow_loading:
                LOGGER.warning(
                    f'## Warning: Hourly {forcings} NetCDF file not found. Falling back to slower csv files.')
        except KeyError:
            fallback_csv = True
            LOGGER.warning(
                f'## Warning: NetCDF file of {forcings} does not contain data for {basin}. Trying slower csv files.')

        if fallback_csv:
            df = load_hourly_us_forcings(self.cfg.data_dir, basin, forcings)

            # Only add hourly discharge if requested.
            if load_discharge:
                df = df.join(load_hourly_us_discharge(self.cfg.data_dir, basin))

        return df


# ==========================================================
# NEW helper: load 15-min QObs (YOU MUST ADAPT THIS FUNCTION)
# ==========================================================
def load_15min_us_discharge(data_dir_15min: Path, basin: str) -> pd.DataFrame:
    """Load 15-min discharge (QObs) for a basin.

    IMPORTANT:
    - I cannot hard-code the exact filename pattern/column name because you haven't shown the file layout yet.
    - You must adapt:
        1) where files are located under /mnt/disk1/CAMELS_US/15min
        2) the filename pattern (e.g., *.csv) and how basin id appears in the name
        3) the datetime column name
        4) the discharge column name (should match YAML target_variables)
    """
    # TODO: Update this pattern to match your 15-min files.
    # Example patterns could be:
    #   pattern = "**/*15min*.csv"
    #   pattern = "**/*usgs-15min.csv"
    pattern = "**/*.csv"

    files = list(data_dir_15min.glob(pattern))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f"No 15-min QObs file found for basin {basin} under {data_dir_15min}")

    # TODO: Update index_col and parse_dates to match your file.
    # Common possibilities: index_col='date' or index_col='time' or index_col='datetime'
    df = pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])

    # TODO: Ensure the column name matches your YAML target_variables.
    # If the file column is qobs_mm_per_hour but your YAML uses QObs(mm/h),
    # rename here so NH sees the expected target name.
    #
    # Example:
    # if 'qobs_mm_per_hour' in df.columns:
    #     df = df.rename(columns={'qobs_mm_per_hour': 'QObs(mm/h)'})

    return df


def load_hourly_us_forcings(data_dir: Path, basin: str, forcings: str) -> pd.DataFrame:
    """Load the hourly forcing data for a basin of the CAMELS US data set."""
    forcing_path = data_dir / 'hourly' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")

    files = list(forcing_path.glob('*.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}')

    # Handle different header types for NLDAS and AORC
    if forcings == 'nldas_hourly':
        df = pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])
    elif forcings == 'aorc_hourly':
        df = pd.read_csv(file_path, index_col='time', parse_dates=['time'])
        df.index = df.index.floor('s')  # Adjust precision to seconds
        df.index.rename('date', inplace=True)  # Rename 'time' to 'date'
    else:
        raise ValueError(f"Unknown forcing type: {forcings}")

    return df


def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the hourly discharge data for a basin of the CAMELS US data set."""
    pattern = '**/*usgs-hourly.csv'
    discharge_path = data_dir / 'hourly' / 'usgs_streamflow'
    files = list(discharge_path.glob(pattern))

    # allow both folder naming variants
    if len(files) == 0:
        discharge_path = discharge_path.parent / 'usgs-streamflow'
        files = list(discharge_path.glob(pattern))

    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {discharge_path}')

    return pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])


def load_hourly_us_stage(data_dir: Path, basin: str) -> pd.Series:
    """Load the hourly stage data for a basin of the CAMELS US data set."""
    stage_path = data_dir / 'hourly' / 'usgs_stage'
    files = list(stage_path.glob('**/*_utc.csv'))
    file_path = [f for f in files if basin in f.stem]
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {stage_path}')

    df = pd.read_csv(file_path,
                     sep=',',
                     index_col=['datetime'],
                     parse_dates=['datetime'],
                     usecols=['datetime', 'gauge_height_ft'])
    df = df.resample('h').mean()
    df["gauge_height_m"] = df["gauge_height_ft"] * 0.3048

    return df["gauge_height_m"]


def load_hourly_us_netcdf(data_dir: Path, forcings: str) -> xarray.Dataset:
    """Load hourly forcing and discharge data from preprocessed netCDF file."""
    netcdf_path = data_dir / 'hourly' / f'usgs-streamflow-{forcings}.nc'
    if not netcdf_path.is_file():
        raise FileNotFoundError(f'No NetCDF file for hourly streamflow and {forcings} at {netcdf_path}.')

    return xarray.open_dataset(netcdf_path)

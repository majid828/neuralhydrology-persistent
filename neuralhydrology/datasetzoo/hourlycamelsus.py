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

    15-min support (QObs only):
    - Dynamic forcings are hourly (AORC/NLDAS). We upsample to 15-min by ffill.
    - Precipitation APCP_surface is an hourly accumulation -> split into four 15-min chunks (divide by 4).
    - Target discharge is read from /mnt/disk1/CAMELS_US/15min/usgs_15min_rdb_by_year as true 15-min QObs.
    """

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: list = [],
                 id_to_int: dict = {},
                 scaler: dict = {}):
        self._netcdf_datasets = {}  # remember datasets for faster reload if present
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
        freqs = getattr(self.cfg, "use_frequencies", None)
        if not freqs:
            return False
        freqs_str = [str(f).lower() for f in freqs]
        return any(("15min" in f) or (f == "15t") or (f == "15") for f in freqs_str)

    # ---------------------------
    # Main loader
    # ---------------------------
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data."""
        run_15min = self._is_15min_run()

        # -------------------------
        # 1) Load FORCINGS (hourly)
        # -------------------------
        dfs = []
        if not any(f.endswith('_hourly') for f in self.cfg.forcings):
            raise ValueError('Forcings include no hourly forcings set.')

        for forcing in self.cfg.forcings:
            if forcing.endswith('_hourly'):
                # In 15-min mode we do NOT need hourly discharge (we load 15-min QObs instead)
                df = self.load_hourly_data(basin, forcing, load_discharge=not run_15min)
            else:
                # Daily forcings -> repeat to hourly
                df, _ = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, forcing)
                df = df.resample('1h').ffill()

            # If multiple forcing products used, rename non-qobs columns to keep uniqueness
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={
                    col: f"{col}_{forcing}" for col in df.columns if 'qobs' not in col.lower()
                })
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # --------------------------------------------
        # 2) Optional: daily discharge QObs(mm/d) feature
        # --------------------------------------------
        all_features = self.cfg.target_variables
        if isinstance(self.cfg.dynamic_inputs, dict):
            for val in self.cfg.dynamic_inputs.values():
                all_features = all_features + val
        elif isinstance(self.cfg.dynamic_inputs, list):
            all_features = all_features + self.cfg.dynamic_inputs_flattened

        if any([x.startswith("QObs(mm/d)") for x in all_features]):
            _, area = camelsus.load_camels_us_forcings(self.cfg.data_dir, basin, "daymet")
            discharge = camelsus.load_camels_us_discharge(self.cfg.data_dir, basin, area)
            discharge = discharge.resample('1h').ffill()
            df["QObs(mm/d)"] = discharge

        self._warn_slow_loading = False

        # ------------------------------------------
        # 3) Clean invalid discharge values (hourly)
        # ------------------------------------------
        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        # -------------------------------------------------
        # 4) Optional: stage
        # -------------------------------------------------
        if 'gauge_height_m' in self.cfg.target_variables:
            df = df.join(load_hourly_us_stage(self.cfg.data_dir, basin))
            df.loc[df['gauge_height_m'] < 0, 'gauge_height_m'] = np.nan

        # ----------------------------------------------------------
        # 5) Optional: synthetic stage
        # ----------------------------------------------------------
        if 'synthetic_qobs_stage_meters' in self.cfg.target_variables:
            attributes = camelsus.load_camels_us_attributes(data_dir=self.cfg.data_dir, basins=[basin])
            with open(self.cfg.rating_curve_file, 'rb') as f:
                rating_curves = pickle.load(f)
            df['synthetic_qobs_stage_meters'] = np.nan
            if basin in rating_curves.keys():
                discharge_m3s = df['qobs_mm_per_hour'].values / 1000 * attributes.area_gages2[basin] * 1e6 / 60**2
                df['synthetic_qobs_stage_meters'] = rating_curves[basin].discharge_to_stage(discharge_m3s)

        # ==========================================================
        # 6) 15-min mode conversion
        # ==========================================================
        if run_15min:
            # Ensure sorted unique hourly index
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # Upsample forcings to 15-min
            df = df.resample("15min").ffill()

            # Split hourly precipitation accumulation into 4 chunks
            if "APCP_surface" in df.columns:
                df["APCP_surface"] = df["APCP_surface"] / 4.0

            # Load TRUE 15-min discharge from RDB
            q15_dir = Path(getattr(self.cfg, "qobs_15min_dir", "/mnt/disk1/CAMELS_US/15min"))
            q15 = load_15min_us_discharge(q15_dir, basin)

            # LEFT JOIN to keep forcing timeline (avoid irregular index)
            df = df.join(q15, how="left")

            # Force regular 15-min grid to avoid infer_frequency(None)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df = df.asfreq("15min")

            # Clean invalid qobs
            if "qobs_cfs" in df.columns:
                df.loc[df["qobs_cfs"] < 0, "qobs_cfs"] = np.nan

        return df

    def load_hourly_data(self, basin: str, forcings: str, load_discharge: bool = True) -> pd.DataFrame:
        """Load a single set of hourly forcings (and optionally hourly discharge)."""
        fallback_csv = False
        try:
            if forcings not in self._netcdf_datasets.keys():
                self._netcdf_datasets[forcings] = load_hourly_us_netcdf(self.cfg.data_dir, forcings)
            df = self._netcdf_datasets[forcings].sel(basin=basin).to_dataframe()
        except FileNotFoundError:
            fallback_csv = True
            if self._warn_slow_loading:
                LOGGER.warning(f'## Warning: Hourly {forcings} NetCDF file not found. Falling back to slower csv files.')
        except KeyError:
            fallback_csv = True
            LOGGER.warning(f'## Warning: NetCDF file of {forcings} does not contain data for {basin}. Trying slower csv files.')

        if fallback_csv:
            df = load_hourly_us_forcings(self.cfg.data_dir, basin, forcings)
            if load_discharge:
                df = df.join(load_hourly_us_discharge(self.cfg.data_dir, basin))

        return df


# ==========================================================
# 15-min USGS QObs loader (RDB)
# ==========================================================
def load_15min_us_discharge(data_dir_15min: Path, basin: str) -> pd.DataFrame:
    """Load 15-min USGS discharge (parameter 00060) from .rdb files.

    Returns a DataFrame indexed by datetime (name='date') with one column:
      - qobs_cfs
    """
    rdb_dir = Path(data_dir_15min) / "usgs_15min_rdb_by_year"
    files = sorted(rdb_dir.glob(f"{basin}_*.rdb"))
    if not files:
        raise FileNotFoundError(f"No 15-min .rdb files found for basin {basin} in {rdb_dir}")

    dfs = []
    for fp in files:
        df = pd.read_csv(
            fp,
            sep="\t",
            comment="#",
            dtype=str,
            engine="python",
            on_bad_lines="skip"
        )
        if df.empty or "datetime" not in df.columns:
            continue

        # Drop the format/width row: "5s 15s 20d 6s ..."
        if str(df["datetime"].iloc[0]).strip().endswith("d"):
            df = df.iloc[1:].copy()

        discharge_cols = [c for c in df.columns if str(c).endswith("_00060")]
        if not discharge_cols:
            continue
        discharge_col = discharge_cols[0]

        df = df[["datetime", discharge_col]].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

        df[discharge_col] = pd.to_numeric(df[discharge_col], errors="coerce")

        df = df.set_index("datetime")
        df.index.name = "date"
        df = df.rename(columns={discharge_col: "qobs_cfs"})

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid discharge (00060) data found for basin {basin} in {rdb_dir}")

    df_all = pd.concat(dfs).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="last")]

    # Force regular 15-min grid (prevents infer_frequency(None))
    full_index = pd.date_range(df_all.index.min(), df_all.index.max(), freq="15min")
    df_all = df_all.reindex(full_index)
    df_all.index.name = "date"

    return df_all


# ==========================================================
# Original hourly CAMELS-US helpers (unchanged)
# ==========================================================
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

    if forcings == 'nldas_hourly':
        df = pd.read_csv(file_path, index_col=['date'], parse_dates=['date'])
    elif forcings == 'aorc_hourly':
        df = pd.read_csv(file_path, index_col='time', parse_dates=['time'])
        df.index = df.index.floor('s')
        df.index.rename('date', inplace=True)
    else:
        raise ValueError(f"Unknown forcing type: {forcings}")

    return df


def load_hourly_us_discharge(data_dir: Path, basin: str) -> pd.DataFrame:
    """Load the hourly discharge data for a basin of the CAMELS US data set."""
    pattern = '**/*usgs-hourly.csv'
    discharge_path = data_dir / 'hourly' / 'usgs_streamflow'
    files = list(discharge_path.glob(pattern))

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

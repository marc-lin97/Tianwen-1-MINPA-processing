# MSE Spatial Grid Statistics

This workflow consumes the existing high-energy daily species products under
`E:\Data\highE` and writes derived spatial statistics to
`E:\Data\highE\processed_mse_spatial_stats`.

## Inputs

- MAVEN STATIC-D1 daily species products:
  `E:\Data\highE\MAVEN\YYYY\maven_static_highE_{Oplus,O2plus}_YYYYMMDD.mat`
- Tianwen-1 MINPA daily species products:
  `E:\Data\highE\Tianwen-1\YYYY\tw1_minpa_highE_{Oplus,O2plus}_YYYYMMDD.mat`

The two missions are processed separately and are never mixed in one grid.

## Grid and Filters

- Coordinate system: MSE.
- Position source fields: `pos_mse_x/y/z_km`.
- Velocity source fields: `v_mse_x/y/z_km_s`.
- Mars radius for this statistical grid: `3397 km`.
- Grid spacing: `0.1 Rm`.
- MAVEN grid centers span `[-3, +3] Rm` on each axis.
- Tianwen-1 grid centers span `[-7, +7] Rm` on each axis.
- Density values below `0.001 cm^-3` are treated as invalid.
- Only rows with `processing_status_code == 1`, finite MSE position, finite FOV
  flag, and valid density enter density statistics.
- Velocity statistics additionally require finite `Vx`, `Vy`, and `Vz`.

## FOV Groups

The products already contain the required MSE `+Z` in-FOV flags:

- MAVEN: `mse_plus_z_in_static_fov_flag`
- Tianwen-1: `mse_plus_z_in_minpa_fov_flag`

For each mission and species, two grids are written:

- `plus_z_in_fov`: flag equals `1`
- `plus_z_out_fov`: flag equals `0`

Rows with `NaN` FOV flags are not included in either group.

## Statistics

Each 3-D grid cell stores:

- `density_count`
- `velocity_count`
- mean and exact median of `density_cm3`
- mean and exact component-wise median of `v_mse_x/y/z_km_s`
- mean and exact median of `speed_mse_km_s = |V_MSE|`

The median is calculated from the records assigned to each 3-D grid cell. The
workflow does not sum densities across records as a local plasma density.
Parallel file chunks only identify the grid cell for each valid record and
return record-level values; the final mean and median are computed after all
files for the same mission, species, and FOV group are combined by grid cell.

## Figures

The first-pass figures are center slices through the full 3-D grids:

- `XY` at the grid cell nearest `Z = 0`
- `XZ` at the grid cell nearest `Y = 0`
- `YZ` at the grid cell nearest `X = 0`

Density panels are plotted as `log10(density)`. Velocity component panels use a
diverging color scale, and speed panels use a sequential color scale.

## Command

```powershell
python scripts\run_mse_spatial_stats.py --workers 4
```

For a quick data-only validation:

```powershell
python scripts\run_mse_spatial_stats.py --workers 2 --no-plots --max-files-per-species 2
```

For a prepared target batch, prefer limiting the scan to the target year or
dates:

```powershell
python scripts\run_mse_spatial_stats.py --years 2021 --workers 4
python scripts\run_mse_spatial_stats.py --dates 20211203 20211204 --workers 4
```

When `--dates` is used, the script builds the expected species file paths
directly inside each mission year folder and does not scan historical
directories.

## Outputs

```text
E:\Data\highE\processed_mse_spatial_stats\data\*.npz
E:\Data\highE\processed_mse_spatial_stats\figures\*.png
E:\Data\highE\processed_mse_spatial_stats\logs\spatial_stats_summary.json
E:\Data\highE\processed_mse_spatial_stats\logs\spatial_stats_summary.csv
```

Grid `.npz` files use sparse storage by default: they contain only non-empty
grid cells plus `axis_x/y/z_rm` arrays and metadata defining the full grid.
Omitted cells have `count = 0` and statistics equal to `NaN` over the declared
MAVEN or Tianwen-1 grid range.

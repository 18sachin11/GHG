import streamlit as st
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from shapely.geometry import mapping
import tempfile
import re


# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="NetCDF Emission Analyzer",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 NetCDF Emission Analyzer using Shapefile Boundary")
st.markdown(
    """
    Upload one or more **NetCDF (.nc)** emission files and a **zipped shapefile** of the study area.
    After clicking **Generate Bar Plot**, the app will clip the NetCDF data using the shapefile
    and calculate total annual emissions for each file.
    """
)


# -------------------------------------------------------
# Constants
# -------------------------------------------------------
SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60
KG_TO_MTON = 1e9          # 1 Mton = 10⁹ kg
TONNES_TO_MTON = 1e6      # 1 Mton = 10⁶ tonnes


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def extract_year_from_filename(filename):
    """
    Extract year from file name.
    Example:
    EDGAR_2025_GHG_GWP_100_AR5_GHG_2004_TOTALS_emi.nc -> 2004
    """
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return int(match.group())
    return filename.replace(".nc", "")


def save_uploaded_file(uploaded_file, output_folder):
    """
    Save uploaded file into temporary directory.
    """
    file_path = Path(output_folder) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_shapefile(shapefile_zip_path):
    """
    Load zipped shapefile using GeoPandas.
    """
    gdf = gpd.read_file(shapefile_zip_path)

    if gdf.empty:
        raise ValueError("The uploaded shapefile is empty.")

    if gdf.crs is None:
        st.warning("Shapefile has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    # Convert shapefile to WGS84 for clipping
    gdf = gdf.to_crs("EPSG:4326")

    return gdf


def standardize_spatial_dims(da):
    """
    Rename spatial dimensions to x and y for rioxarray.
    """
    rename_dict = {}

    if "lon" in da.dims:
        rename_dict["lon"] = "x"
    if "longitude" in da.dims:
        rename_dict["longitude"] = "x"
    if "X" in da.dims:
        rename_dict["X"] = "x"

    if "lat" in da.dims:
        rename_dict["lat"] = "y"
    if "latitude" in da.dims:
        rename_dict["latitude"] = "y"
    if "Y" in da.dims:
        rename_dict["Y"] = "y"

    if rename_dict:
        da = da.rename(rename_dict)

    if "x" not in da.dims or "y" not in da.dims:
        raise ValueError(
            f"Could not identify spatial dimensions. Found dimensions: {da.dims}"
        )

    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    return da


def reduce_extra_dimensions(da):
    """
    If the DataArray contains dimensions other than x and y,
    select the first available slice from those dimensions.
    """
    extra_dims = [dim for dim in da.dims if dim not in ["x", "y"]]

    for dim in extra_dims:
        da = da.isel({dim: 0})

    return da.squeeze()


def assign_crs_to_dataarray(da, ds):
    """
    Assign CRS to NetCDF DataArray.
    Default CRS is EPSG:4326 if not available.
    """
    try:
        if da.rio.crs is None:
            if "crs" in ds.attrs:
                da = da.rio.write_crs(ds.attrs["crs"], inplace=False)
            elif "spatial_ref" in ds:
                try:
                    da = da.rio.write_crs(ds["spatial_ref"].attrs.get("crs_wkt"), inplace=False)
                except Exception:
                    da = da.rio.write_crs("EPSG:4326", inplace=False)
            else:
                da = da.rio.write_crs("EPSG:4326", inplace=False)
    except Exception:
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    return da


def calculate_total_emissions(clipped_da, emission_unit, target_crs):
    """
    Calculate total emissions in Mton/year.
    """
    clipped_projected = clipped_da.rio.reproject(target_crs)

    x_res = abs(clipped_projected.rio.resolution()[0])
    y_res = abs(clipped_projected.rio.resolution()[1])
    grid_cell_area = x_res * y_res

    unit = str(emission_unit).strip().lower()

    if unit in [
        "kg / (m2 * s)",
        "kg m-2 s-1",
        "kg/m2/s",
        "kg m^-2 s^-1",
        "kg m-2 sec-1"
    ]:
        total_mton = (
            clipped_projected * grid_cell_area * SECONDS_IN_YEAR / KG_TO_MTON
        ).sum(skipna=True).item()

    elif unit in [
        "kg / m2 / yr",
        "kg m-2 yr-1",
        "kg/m2/yr",
        "kg m^-2 yr^-1",
        "kg m-2 year-1"
    ]:
        total_mton = (
            clipped_projected * grid_cell_area / KG_TO_MTON
        ).sum(skipna=True).item()

    elif unit in [
        "kg",
        "kg/year",
        "kg/yr",
        "kg yr-1"
    ]:
        total_mton = clipped_projected.sum(skipna=True).item() / KG_TO_MTON

    elif unit in [
        "tonnes",
        "tonne",
        "tons",
        "ton",
        "t"
    ]:
        total_mton = clipped_projected.sum(skipna=True).item() / TONNES_TO_MTON

    elif unit in [
        "mton",
        "mton/year",
        "mton/yr"
    ]:
        total_mton = clipped_projected.sum(skipna=True).item()

    else:
        st.warning(
            f"Unrecognized unit '{emission_unit}'. "
            "Assuming kg / (m² × s)."
        )
        total_mton = (
            clipped_projected * grid_cell_area * SECONDS_IN_YEAR / KG_TO_MTON
        ).sum(skipna=True).item()

    return total_mton, grid_cell_area


def process_netcdf_file(nc_path, gdf, variable_name, target_crs):
    """
    Process one NetCDF file:
    - open dataset
    - select emissions variable
    - standardize dimensions
    - assign CRS
    - clip with shapefile
    - calculate total emissions
    """
    ds = xr.open_dataset(nc_path)

    if variable_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{variable_name}' not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable_name].squeeze()

    da = standardize_spatial_dims(da)
    da = reduce_extra_dimensions(da)
    da = assign_crs_to_dataarray(da, ds)

    if da.rio.crs != "EPSG:4326":
        da = da.rio.reproject("EPSG:4326")

    clipped_da = da.rio.clip(
        gdf.geometry.apply(mapping),
        gdf.crs,
        drop=True
    )

    if clipped_da.size == 0:
        raise ValueError("No NetCDF data found inside the shapefile boundary.")

    emission_unit = ds[variable_name].attrs.get("units", "unknown")

    total_mton, grid_cell_area = calculate_total_emissions(
        clipped_da,
        emission_unit,
        target_crs
    )

    return total_mton, emission_unit, grid_cell_area


# -------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------
st.sidebar.header("⚙️ Input Settings")

uploaded_nc_files = st.sidebar.file_uploader(
    "Upload NetCDF files",
    type=["nc"],
    accept_multiple_files=True
)

uploaded_shapefile = st.sidebar.file_uploader(
    "Upload study area shapefile as ZIP",
    type=["zip"]
)

variable_name = st.sidebar.text_input(
    "NetCDF variable name",
    value="emissions"
)

target_crs = st.sidebar.text_input(
    "Projected CRS for area calculation",
    value="EPSG:32644",
    help="Use a suitable projected CRS for your study area. For Jammu/Ladakh/parts of NW India, EPSG:32643 or EPSG:32644 may be suitable."
)

generate = st.sidebar.button("Generate Bar Plot", type="primary")


# -------------------------------------------------------
# Main Processing
# -------------------------------------------------------
if generate:

    if not uploaded_nc_files:
        st.error("Please upload at least one NetCDF file.")
        st.stop()

    if uploaded_shapefile is None:
        st.error("Please upload a zipped shapefile.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:

        try:
            shapefile_path = save_uploaded_file(uploaded_shapefile, temp_dir)
            gdf = load_shapefile(shapefile_path)

            st.success("Shapefile loaded successfully.")
            st.write(f"Shapefile CRS after reprojection: `{gdf.crs}`")
            st.write(f"Number of features in shapefile: `{len(gdf)}`")

        except Exception as e:
            st.error(f"Error loading shapefile: {e}")
            st.stop()

        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, nc_file in enumerate(uploaded_nc_files):

            year_or_name = extract_year_from_filename(nc_file.name)
            status_text.info(f"Processing: {nc_file.name}")

            try:
                nc_path = save_uploaded_file(nc_file, temp_dir)

                total_mton, emission_unit, grid_cell_area = process_netcdf_file(
                    nc_path=nc_path,
                    gdf=gdf,
                    variable_name=variable_name,
                    target_crs=target_crs
                )

                results.append({
                    "File": nc_file.name,
                    "Year": year_or_name,
                    "Emission Unit": emission_unit,
                    "Grid Cell Area (m²)": round(grid_cell_area, 2),
                    "Total Emissions (Mton/year)": round(total_mton, 4)
                })

            except Exception as e:
                results.append({
                    "File": nc_file.name,
                    "Year": year_or_name,
                    "Emission Unit": "Error",
                    "Grid Cell Area (m²)": np.nan,
                    "Total Emissions (Mton/year)": np.nan,
                    "Remarks": str(e)
                })

            progress_bar.progress((i + 1) / len(uploaded_nc_files))

        status_text.success("Processing completed.")

        result_df = pd.DataFrame(results)

        st.subheader("📋 Emission Calculation Results")
        st.dataframe(result_df, use_container_width=True)

        valid_df = result_df.dropna(subset=["Total Emissions (Mton/year)"])

        if valid_df.empty:
            st.error("No valid emission values were calculated.")
            st.stop()

        # Sort by year if possible
        try:
            valid_df = valid_df.sort_values("Year")
        except Exception:
            pass

        # -------------------------------------------------------
        # Bar Plot
        # -------------------------------------------------------
        st.subheader("📊 Annual Emission Bar Plot")

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            valid_df["Year"].astype(str),
            valid_df["Total Emissions (Mton/year)"]
        )

        ax.set_xlabel("Year / File")
        ax.set_ylabel("Total Emissions (Mton/year)")
        ax.set_title("Total Annual Emissions within Uploaded Shapefile Boundary")

        ax.tick_params(axis="x", rotation=45)

        # Add values above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.tight_layout()
        st.pyplot(fig)

        # -------------------------------------------------------
        # Download Result CSV
        # -------------------------------------------------------
        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="emission_results.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Upload NetCDF files and a zipped shapefile from the sidebar, "
        "then click **Generate Bar Plot**."
    )

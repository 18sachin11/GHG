import streamlit as st
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import tempfile
import re
import traceback


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="NetCDF Emission Bar Plot Tool",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 NetCDF Emission Analysis Tool")
st.markdown(
    """
    Upload one or more **NetCDF files** and a **zipped shapefile** of the study area.
    The app will clip the NetCDF emission data over the shapefile boundary and generate
    a bar plot with emission values.
    """
)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------
SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60
KG_TO_MTON = 1e9
TONNES_TO_MTON = 1e6
EARTH_RADIUS = 6371000  # metres


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def save_uploaded_file(uploaded_file, folder):
    file_path = Path(folder) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_year_from_filename(filename):
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return int(match.group())
    return filename.replace(".nc", "")


def load_zipped_shapefile(zip_path):
    gdf = gpd.read_file(zip_path)

    if gdf.empty:
        raise ValueError("The uploaded shapefile is empty.")

    if gdf.crs is None:
        st.warning("The shapefile has no CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]

    if gdf.empty:
        raise ValueError("No valid geometry found in the shapefile.")

    return gdf


def get_lat_lon_names(ds):
    possible_lon_names = ["lon", "longitude", "x", "X", "LONGITUDE", "LON"]
    possible_lat_names = ["lat", "latitude", "y", "Y", "LATITUDE", "LAT"]

    lon_name = None
    lat_name = None

    for name in possible_lon_names:
        if name in ds.coords or name in ds.dims:
            lon_name = name
            break

    for name in possible_lat_names:
        if name in ds.coords or name in ds.dims:
            lat_name = name
            break

    if lon_name is None or lat_name is None:
        raise ValueError(
            f"Could not detect latitude and longitude names. "
            f"Available coordinates: {list(ds.coords)}; dimensions: {list(ds.dims)}"
        )

    return lat_name, lon_name


def reduce_extra_dimensions(da, lat_name, lon_name):
    extra_dims = [d for d in da.dims if d not in [lat_name, lon_name]]

    for dim in extra_dims:
        da = da.isel({dim: 0})

    return da.squeeze()


def fix_longitude_if_needed(da, lon_name, gdf):
    lon_values = da[lon_name].values

    # Convert 0–360 longitude to -180–180 if shapefile uses negative longitudes
    if np.nanmax(lon_values) > 180:
        new_lon = ((lon_values + 180) % 360) - 180
        da = da.assign_coords({lon_name: new_lon})
        da = da.sortby(lon_name)

    return da


def subset_to_shapefile_bounds(da, lat_name, lon_name, gdf):
    minx, miny, maxx, maxy = gdf.total_bounds

    lon_values = da[lon_name].values
    lat_values = da[lat_name].values

    lon_res = abs(np.nanmedian(np.diff(np.sort(lon_values))))
    lat_res = abs(np.nanmedian(np.diff(np.sort(lat_values))))

    # Ensure longitude is ascending
    da = da.sortby(lon_name)

    # Ensure latitude is descending for raster-style masking
    da = da.sortby(lat_name, ascending=False)

    minx = minx - lon_res
    maxx = maxx + lon_res
    miny = miny - lat_res
    maxy = maxy + lat_res

    da_sub = da.sel(
        {
            lon_name: slice(minx, maxx),
            lat_name: slice(maxy, miny)
        }
    )

    if da_sub.size == 0:
        raise ValueError(
            "No NetCDF grid cells overlap with the uploaded shapefile. "
            "Check whether the NetCDF and shapefile cover the same region."
        )

    return da_sub


def create_geometry_mask(da, lat_name, lon_name, gdf):
    lon = da[lon_name].values
    lat = da[lat_name].values

    if len(lon) < 2 or len(lat) < 2:
        raise ValueError("Latitude or longitude dimension has too few values.")

    width = len(lon)
    height = len(lat)

    lon_res = abs(np.nanmedian(np.diff(lon)))
    lat_res = abs(np.nanmedian(np.diff(lat)))

    west = np.nanmin(lon) - lon_res / 2
    east = np.nanmax(lon) + lon_res / 2
    south = np.nanmin(lat) - lat_res / 2
    north = np.nanmax(lat) + lat_res / 2

    transform = from_bounds(
        west,
        south,
        east,
        north,
        width,
        height
    )

    try:
        geom = gdf.geometry.union_all()
    except Exception:
        geom = gdf.geometry.unary_union

    mask = geometry_mask(
        [mapping(geom)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=True
    )

    return mask


def calculate_grid_cell_area(lat_values, lon_values):
    """
    Calculates area of each lat-lon grid cell in m².
    Area varies with latitude.
    """
    lat_values = np.asarray(lat_values)
    lon_values = np.asarray(lon_values)

    lat_res = abs(np.nanmedian(np.diff(lat_values)))
    lon_res = abs(np.nanmedian(np.diff(lon_values)))

    lat_upper = np.radians(lat_values + lat_res / 2)
    lat_lower = np.radians(lat_values - lat_res / 2)
    lon_res_rad = np.radians(lon_res)

    row_area = (
        EARTH_RADIUS ** 2
        * lon_res_rad
        * np.abs(np.sin(lat_upper) - np.sin(lat_lower))
    )

    area_2d = np.repeat(row_area[:, np.newaxis], len(lon_values), axis=1)

    return area_2d


def calculate_total_emission(data_values, area_2d, mask, unit):
    unit = str(unit).strip().lower()

    masked_values = np.where(mask, data_values, np.nan)

    if unit in [
        "kg / (m2 * s)",
        "kg m-2 s-1",
        "kg/m2/s",
        "kg m^-2 s^-1",
        "kg m-2 sec-1"
    ]:
        total_mton = np.nansum(masked_values * area_2d * SECONDS_IN_YEAR) / KG_TO_MTON

    elif unit in [
        "kg / m2 / yr",
        "kg m-2 yr-1",
        "kg/m2/yr",
        "kg m^-2 yr^-1",
        "kg m-2 year-1"
    ]:
        total_mton = np.nansum(masked_values * area_2d) / KG_TO_MTON

    elif unit in [
        "kg",
        "kg/year",
        "kg/yr",
        "kg yr-1"
    ]:
        total_mton = np.nansum(masked_values) / KG_TO_MTON

    elif unit in [
        "tonnes",
        "tonne",
        "tons",
        "ton",
        "t"
    ]:
        total_mton = np.nansum(masked_values) / TONNES_TO_MTON

    elif unit in [
        "mton",
        "mton/year",
        "mton/yr"
    ]:
        total_mton = np.nansum(masked_values)

    else:
        st.warning(
            f"Unknown unit: {unit}. Assuming kg m⁻² s⁻¹."
        )
        total_mton = np.nansum(masked_values * area_2d * SECONDS_IN_YEAR) / KG_TO_MTON

    return total_mton


def process_single_nc(nc_path, gdf, variable_name):
    ds = xr.open_dataset(nc_path)

    if variable_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{variable_name}' not found in NetCDF file. "
            f"Available variables are: {list(ds.data_vars)}"
        )

    lat_name, lon_name = get_lat_lon_names(ds)

    da = ds[variable_name]
    da = reduce_extra_dimensions(da, lat_name, lon_name)
    da = fix_longitude_if_needed(da, lon_name, gdf)
    da = subset_to_shapefile_bounds(da, lat_name, lon_name, gdf)

    # Keep only lat-lon order
    da = da.transpose(lat_name, lon_name)

    mask = create_geometry_mask(da, lat_name, lon_name, gdf)

    values = da.values.astype(float)

    lat_values = da[lat_name].values
    lon_values = da[lon_name].values

    area_2d = calculate_grid_cell_area(lat_values, lon_values)

    unit = ds[variable_name].attrs.get("units", "unknown")

    total_mton = calculate_total_emission(
        data_values=values,
        area_2d=area_2d,
        mask=mask,
        unit=unit
    )

    valid_pixel_count = int(np.sum(mask))

    return total_mton, unit, valid_pixel_count


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("Input Data")

uploaded_nc_files = st.sidebar.file_uploader(
    "Upload NetCDF file/files",
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

generate = st.sidebar.button("Generate Bar Plot", type="primary")


# ---------------------------------------------------
# Main App Logic
# ---------------------------------------------------
if generate:

    if uploaded_shapefile is None:
        st.error("Please upload the study area shapefile as a ZIP file.")
        st.stop()

    if not uploaded_nc_files:
        st.error("Please upload at least one NetCDF file.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:

        try:
            shapefile_zip_path = save_uploaded_file(uploaded_shapefile, temp_dir)
            gdf = load_zipped_shapefile(shapefile_zip_path)

            st.success("Shapefile loaded successfully.")
            st.write(f"Detected shapefile CRS after conversion: `{gdf.crs}`")
            st.write(f"Number of shapefile features: `{len(gdf)}`")

        except Exception as e:
            st.error("Error while reading shapefile.")
            st.code(str(e))
            st.stop()

        results = []

        progress = st.progress(0)
        status = st.empty()

        for i, nc_file in enumerate(uploaded_nc_files):

            status.info(f"Processing {nc_file.name}")

            try:
                nc_path = save_uploaded_file(nc_file, temp_dir)

                year = extract_year_from_filename(nc_file.name)

                total_mton, unit, valid_pixels = process_single_nc(
                    nc_path=nc_path,
                    gdf=gdf,
                    variable_name=variable_name
                )

                results.append(
                    {
                        "File": nc_file.name,
                        "Year": year,
                        "Unit": unit,
                        "Valid Pixels inside AOI": valid_pixels,
                        "Total Emission (Mton/year)": total_mton,
                        "Status": "Success"
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "File": nc_file.name,
                        "Year": extract_year_from_filename(nc_file.name),
                        "Unit": "Error",
                        "Valid Pixels inside AOI": 0,
                        "Total Emission (Mton/year)": np.nan,
                        "Status": str(e)
                    }
                )

                with st.expander(f"Error details for {nc_file.name}"):
                    st.code(traceback.format_exc())

            progress.progress((i + 1) / len(uploaded_nc_files))

        status.success("Processing completed.")

        result_df = pd.DataFrame(results)

        st.subheader("Calculated Emission Results")
        st.dataframe(result_df, use_container_width=True)

        valid_df = result_df.dropna(subset=["Total Emission (Mton/year)"])

        if valid_df.empty:
            st.error("No valid emission result was generated. Please check the error details above.")
            st.stop()

        try:
            valid_df = valid_df.sort_values("Year")
        except Exception:
            pass

        # ---------------------------------------------------
        # Bar Plot
        # ---------------------------------------------------
        st.subheader("Annual Emission Bar Plot")

        fig, ax = plt.subplots(figsize=(11, 6))

        bars = ax.bar(
            valid_df["Year"].astype(str),
            valid_df["Total Emission (Mton/year)"]
        )

        ax.set_xlabel("Year / File")
        ax.set_ylabel("Total Emission (Mton/year)")
        ax.set_title("Total Annual Emission within Uploaded Shapefile Boundary")

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

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        # ---------------------------------------------------
        # CSV Download
        # ---------------------------------------------------
        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name="emission_results.csv",
            mime="text/csv"
        )

else:
    st.info(
        "Upload NetCDF file/files and a zipped shapefile from the sidebar, then click Generate Bar Plot."
    )

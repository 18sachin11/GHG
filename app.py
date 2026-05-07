import streaml:contentReference[oaicite:1]{index=1}das as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from shapely.ops import unary_union
from shapely import points, covers
import tempfile
import zipfile
import re
import traceback


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GHG Emission Analyzer",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GHG Emission Analyzer from NetCDF")
st.markdown(
    """
    This app calculates total greenhouse gas emissions within an uploaded study-area boundary.
    
    Upload:
    
    1. One or more **NetCDF (.nc)** files  
    2. One **zipped shapefile (.zip)** of the study area  
    
    Then click **Generate Bar Plot**.
    """
)


# ============================================================
# CONSTANTS
# ============================================================

SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60
KG_TO_MTON = 1e9
TONNES_TO_MTON = 1e6
EARTH_RADIUS = 6371000  # metres


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_uploaded_file(uploaded_file, folder):
    file_path = Path(folder) / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_year_from_filename(filename):
    """
    Extracts year from filename.
    Example:
    EDGAR_2025_GHG_GWP_100_AR5_GHG_2004_TOTALS_emi.nc -> 2004
    """
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        return int(match.group())

    return filename.replace(".nc", "")


def extract_and_read_shapefile(zip_path, temp_dir):
    """
    Extract zipped shapefile and read it using pyogrio engine.
    This avoids Fiona/GDAL build errors on Streamlit Cloud.
    """

    extract_dir = Path(temp_dir) / "uploaded_shapefile"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    shp_files = list(extract_dir.rglob("*.shp"))

    if len(shp_files) == 0:
        raise ValueError(
            "No .shp file found inside the uploaded ZIP. "
            "Please upload a complete zipped shapefile containing .shp, .shx, .dbf and .prj files."
        )

    shp_path = shp_files[0]

    gdf = gpd.read_file(shp_path, engine="pyogrio")

    if gdf.empty:
        raise ValueError("The uploaded shapefile is empty.")

    if gdf.crs is None:
        st.warning("The shapefile has no CRS. EPSG:4326 is assumed.")
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]

    if gdf.empty:
        raise ValueError("No valid geometry found in the shapefile.")

    return gdf


def get_lat_lon_names(ds):
    """
    Detect latitude and longitude coordinate names.
    """

    possible_lon_names = [
        "lon", "longitude", "Longitude", "LONGITUDE",
        "x", "X"
    ]

    possible_lat_names = [
        "lat", "latitude", "Latitude", "LATITUDE",
        "y", "Y"
    ]

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
            "Could not automatically detect latitude and longitude. "
            f"Available coordinates: {list(ds.coords)}. "
            f"Available dimensions: {list(ds.dims)}."
        )

    return lat_name, lon_name


def reduce_extra_dimensions(da, lat_name, lon_name):
    """
    Removes non-spatial dimensions by selecting the first slice.
    Example: time, band, level, etc.
    """

    extra_dims = [dim for dim in da.dims if dim not in [lat_name, lon_name]]

    if len(extra_dims) > 0:
        st.warning(
            f"Extra dimension(s) found: {extra_dims}. "
            "The first slice of each extra dimension will be used."
        )

    for dim in extra_dims:
        da = da.isel({dim: 0})

    return da.squeeze()


def normalize_longitude(da, lon_name, gdf):
    """
    Converts 0–360 longitude to -180 to 180 if required.
    """

    lon_values = da[lon_name].values

    if np.nanmax(lon_values) > 180:
        new_lon = ((lon_values + 180) % 360) - 180
        da = da.assign_coords({lon_name: new_lon})
        da = da.sortby(lon_name)

    return da


def subset_to_aoi_bounds(da, lat_name, lon_name, gdf):
    """
    Subset NetCDF grid to shapefile bounding box.
    """

    minx, miny, maxx, maxy = gdf.total_bounds

    lon_values = da[lon_name].values
    lat_values = da[lat_name].values

    lon_res = abs(np.nanmedian(np.diff(np.sort(lon_values))))
    lat_res = abs(np.nanmedian(np.diff(np.sort(lat_values))))

    da = da.sortby(lon_name)
    da = da.sortby(lat_name)

    minx = minx - lon_res
    maxx = maxx + lon_res
    miny = miny - lat_res
    maxy = maxy + lat_res

    da_sub = da.sel(
        {
            lon_name: slice(minx, maxx),
            lat_name: slice(miny, maxy)
        }
    )

    if da_sub.size == 0:
        raise ValueError(
            "No NetCDF grid cells overlap with the uploaded shapefile. "
            "Please check whether both datasets cover the same region."
        )

    return da_sub


def create_aoi_mask(da, lat_name, lon_name, gdf):
    """
    Creates a spatial mask using shapely vectorized points.
    True means the grid-cell centre lies inside the AOI.
    """

    lat_values = da[lat_name].values
    lon_values = da[lon_name].values

    lon_2d, lat_2d = np.meshgrid(lon_values, lat_values)

    geom = unary_union(gdf.geometry)

    point_array = points(lon_2d.ravel(), lat_2d.ravel())

    mask_flat = covers(geom, point_array)

    mask = np.asarray(mask_flat).reshape(lon_2d.shape)

    return mask


def calculate_grid_cell_area(lat_values, lon_values):
    """
    Calculates area of each latitude-longitude grid cell in square metres.
    The area changes with latitude.
    """

    lat_values = np.asarray(lat_values)
    lon_values = np.asarray(lon_values)

    lat_res = abs(np.nanmedian(np.diff(np.sort(lat_values))))
    lon_res = abs(np.nanmedian(np.diff(np.sort(lon_values))))

    lat_upper = np.radians(lat_values + lat_res / 2)
    lat_lower = np.radians(lat_values - lat_res / 2)

    lon_res_rad = np.radians(lon_res)

    row_area = (
        EARTH_RADIUS ** 2
        * lon_res_rad
        * np.abs(np.sin(lat_upper) - np.sin(lat_lower))
    )

    area_2d = np.repeat(
        row_area[:, np.newaxis],
        len(lon_values),
        axis=1
    )

    return area_2d


def calculate_total_emission(values, area_2d, mask, unit):
    """
    Converts emissions to Mton/year depending on unit.
    """

    unit_original = str(unit)
    unit_clean = unit_original.strip().lower()

    masked_values = np.where(mask, values, np.nan)

    if unit_clean in [
        "kg / (m2 * s)",
        "kg m-2 s-1",
        "kg/m2/s",
        "kg m^-2 s^-1",
        "kg m-2 sec-1",
        "kg m**-2 s**-1"
    ]:
        total_mton = (
            np.nansum(masked_values * area_2d * SECONDS_IN_YEAR)
            / KG_TO_MTON
        )

    elif unit_clean in [
        "kg / m2 / yr",
        "kg m-2 yr-1",
        "kg/m2/yr",
        "kg m^-2 yr^-1",
        "kg m-2 year-1",
        "kg m**-2 yr**-1"
    ]:
        total_mton = (
            np.nansum(masked_values * area_2d)
            / KG_TO_MTON
        )

    elif unit_clean in [
        "kg",
        "kg/year",
        "kg/yr",
        "kg yr-1"
    ]:
        total_mton = np.nansum(masked_values) / KG_TO_MTON

    elif unit_clean in [
        "tonnes",
        "tonne",
        "tons",
        "ton",
        "t"
    ]:
        total_mton = np.nansum(masked_values) / TONNES_TO_MTON

    elif unit_clean in [
        "mton",
        "mton/year",
        "mton/yr",
        "million tonnes",
        "million tonnes/year"
    ]:
        total_mton = np.nansum(masked_values)

    else:
        st.warning(
            f"Unknown emission unit: '{unit_original}'. "
            "The app assumes kg m⁻² s⁻¹."
        )

        total_mton = (
            np.nansum(masked_values * area_2d * SECONDS_IN_YEAR)
            / KG_TO_MTON
        )

    return total_mton


def process_netcdf_file(nc_path, gdf, variable_name):
    """
    Main processing function for one NetCDF file.
    """

    ds = xr.open_dataset(nc_path)

    if variable_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{variable_name}' not found. "
            f"Available variables are: {list(ds.data_vars)}"
        )

    lat_name, lon_name = get_lat_lon_names(ds)

    da = ds[variable_name]

    da = reduce_extra_dimensions(da, lat_name, lon_name)
    da = normalize_longitude(da, lon_name, gdf)
    da = subset_to_aoi_bounds(da, lat_name, lon_name, gdf)

    da = da.transpose(lat_name, lon_name)

    values = da.values.astype(float)

    lat_values = da[lat_name].values
    lon_values = da[lon_name].values

    mask = create_aoi_mask(da, lat_name, lon_name, gdf)

    if np.sum(mask) == 0:
        raise ValueError(
            "The shapefile overlaps the NetCDF bounding box, "
            "but no grid-cell centres fall inside the polygon."
        )

    area_2d = calculate_grid_cell_area(lat_values, lon_values)

    unit = ds[variable_name].attrs.get("units", "unknown")

    total_mton = calculate_total_emission(
        values=values,
        area_2d=area_2d,
        mask=mask,
        unit=unit
    )

    valid_pixels = int(np.sum(mask))

    return total_mton, unit, valid_pixels, lat_name, lon_name


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("📂 Upload Data")

uploaded_nc_files = st.sidebar.file_uploader(
    "Upload NetCDF file/files",
    type=["nc"],
    accept_multiple_files=True
)

uploaded_shapefile = st.sidebar.file_uploader(
    "Upload study-area shapefile as ZIP",
    type=["zip"]
)

variable_name = st.sidebar.text_input(
    "NetCDF variable name",
    value="emissions"
)

st.sidebar.markdown("---")

generate = st.sidebar.button(
    "Generate Bar Plot",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN APP
# ============================================================

if generate:

    if uploaded_shapefile is None:
        st.error("Please upload the study-area shapefile as a ZIP file.")
        st.stop()

    if not uploaded_nc_files:
        st.error("Please upload at least one NetCDF file.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:

        # ----------------------------------------------------
        # Read shapefile
        # ----------------------------------------------------
        try:
            shapefile_zip_path = save_uploaded_file(
                uploaded_shapefile,
                temp_dir
            )

            gdf = extract_and_read_shapefile(
                shapefile_zip_path,
                temp_dir
            )

            st.success("Shapefile loaded successfully.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Number of Features", len(gdf))

            with col2:
                st.metric("CRS", str(gdf.crs))

            with col3:
                minx, miny, maxx, maxy = gdf.total_bounds
                st.metric(
                    "AOI Bounds",
                    f"{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}"
                )

        except Exception as e:
            st.error("Error while reading the shapefile.")
            st.code(str(e))
            st.stop()

        # ----------------------------------------------------
        # Process NetCDF files
        # ----------------------------------------------------
        results = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, nc_file in enumerate(uploaded_nc_files):

            status_text.info(f"Processing: {nc_file.name}")

            try:
                nc_path = save_uploaded_file(nc_file, temp_dir)

                year = extract_year_from_filename(nc_file.name)

                total_mton, unit, valid_pixels, lat_name, lon_name = process_netcdf_file(
                    nc_path=nc_path,
                    gdf=gdf,
                    variable_name=variable_name
                )

                results.append(
                    {
                        "File": nc_file.name,
                        "Year": year,
                        "Variable": variable_name,
                        "Unit": unit,
                        "Latitude Field": lat_name,
                        "Longitude Field": lon_name,
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
                        "Variable": variable_name,
                        "Unit": "Error",
                        "Latitude Field": "-",
                        "Longitude Field": "-",
                        "Valid Pixels inside AOI": 0,
                        "Total Emission (Mton/year)": np.nan,
                        "Status": str(e)
                    }
                )

                with st.expander(f"Error details for {nc_file.name}"):
                    st.code(traceback.format_exc())

            progress_bar.progress((i + 1) / len(uploaded_nc_files))

        status_text.success("Processing completed.")

        result_df = pd.DataFrame(results)

        # ----------------------------------------------------
        # Display results table
        # ----------------------------------------------------
        st.subheader("📋 Calculated Emission Results")

        display_df = result_df.copy()

        if "Total Emission (Mton/year)" in display_df.columns:
            display_df["Total Emission (Mton/year)"] = display_df[
                "Total Emission (Mton/year)"
            ].round(4)

        st.dataframe(display_df, use_container_width=True)

        valid_df = result_df.dropna(
            subset=["Total Emission (Mton/year)"]
        ).copy()

        if valid_df.empty:
            st.error(
                "No valid emission result was generated. "
                "Please check the error details shown above."
            )
            st.stop()

        try:
            valid_df = valid_df.sort_values("Year")
        except Exception:
            pass

        # ----------------------------------------------------
        # Bar plot
        # ----------------------------------------------------
        st.subheader("📊 Annual Emission Bar Plot")

        fig, ax = plt.subplots(figsize=(11, 6))

        x_labels = valid_df["Year"].astype(str)
        y_values = valid_df["Total Emission (Mton/year)"]

        bars = ax.bar(x_labels, y_values)

        ax.set_xlabel("Year / File", fontsize=12)
        ax.set_ylabel("Total Emission (Mton/year)", fontsize=12)
        ax.set_title(
            "Total Annual GHG Emission within Uploaded Study Area",
            fontsize=14,
            fontweight="bold"
        )

        ax.grid(axis="y", linestyle="--", alpha=0.4)

        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------
        st.subheader("🧾 Summary")

        max_row = valid_df.loc[
            valid_df["Total Emission (Mton/year)"].idxmax()
        ]

        min_row = valid_df.loc[
            valid_df["Total Emission (Mton/year)"].idxmin()
        ]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Maximum Emission",
                f"{max_row['Total Emission (Mton/year)']:.2f} Mton/year",
                f"Year/File: {max_row['Year']}"
            )

        with col2:
            st.metric(
                "Minimum Emission",
                f"{min_row['Total Emission (Mton/year)']:.2f} Mton/year",
                f"Year/File: {min_row['Year']}"
            )

        with col3:
            st.metric(
                "Files Successfully Processed",
                len(valid_df)
            )

        # ----------------------------------------------------
        # Download CSV
        # ----------------------------------------------------
        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name="ghg_emission_results.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info(
        "Upload NetCDF file/files and a zipped shapefile from the sidebar, "
        "then click **Generate Bar Plot**."
    )

    st.markdown(
        """
        ### Required shapefile ZIP contents
        
        Your ZIP file should contain:
        
        ```text
        study_area.shp
        study_area.shx
        study_area.dbf
        study_area.prj
        ```
        
        ### Default NetCDF variable name
        
        The default variable name is:
        
        ```text
        emissions
        ```
        
        Change it in the sidebar if your NetCDF file uses another variable name.
        """
    )

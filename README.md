# JWST NIRSpec Pathloss Model

This script models the pathloss for the James Webb Space Telescope (JWST) Near-Infrared Spectrograph (NIRSpec) instrument. The pathloss is the fraction of light from a point source that is lost due to the finite size of the spectrograph slit. This model is based on the error function (erf), which is derived from integrating a Gaussian Point Spread Function (PSF) over a rectangular slit.

## Features

*   **Physical Modeling:** Implements a 2D physical model for PSF throughput through a rectangular slit using the error function.
*   **Data Fitting:** Fits the model to observational data to determine key parameters like slit dimensions and PSF size as a function of wavelength.
*   **Uncertainty Analysis:** Provides a detailed uncertainty analysis, including the calculation of residuals, 2D scatter, and a smoothed, monotonic uncertainty model.
*   **Data Visualization:** Generates a comprehensive set of plots to visualize the data, model fits, residuals, and uncertainty characterization.
*   **FITS File Generation:** Creates and updates FITS files with the modeled pathloss and its associated variance, suitable for use in astronomical data analysis pipelines.
*   **Command-Line Interface:** Offers a flexible command-line interface to control the script's execution, including plotting, prediction, and FITS file updates.

## Installation

This script requires the following Python libraries:

*   `numpy`
*   `matplotlib`
*   `astropy`
*   `scipy`

You can install these dependencies using pip:

```bash
pip install numpy matplotlib astropy scipy
```

## Usage

The script `model_pathloss_data.py` can be run from the command line with various options.

### Basic Execution

To run the full analysis, including fitting the model to the data and generating all plots, simply run the script without any arguments:

```bash
python model_pathloss_data.py
```

This will create a new directory (e.g., `plots_throughput_model_sparse1`) containing all the output plots.

### Command-Line Arguments

*   `--plot-arrays <FITS_FILE>`: Plot the pathloss arrays from a given FITS file.
    ```bash
    python model_pathloss_data.py --plot-arrays jwst_nirspec_pathloss_erf.fits
    ```

*   `--summary-only`: Only generate the final summary plots (parameters vs. wavelength and scatter summary).
    ```bash
    python model_pathloss_data.py --summary-only
    ```

*   `--predict-wavelength <WAVELENGTH>`: Predict the pathloss for a specific wavelength and save it to a FITS file.
    ```bash
    python model_pathloss_data.py --predict-wavelength 2.5 --output-fits prediction.fits
    ```

*   `--update-fits`: Update the `jwst_nirspec_pathloss_0005.fits` file with the model-derived pathloss and variance, saving the result to `jwst_nirspec_pathloss_erf.fits`.
    ```bash
    python model_pathloss_data.py --update-fits
    ```

*   `--fit-slit-dims-once`: Fit the slit dimensions for the first wavelength only and keep them constant for all other wavelengths.
    ```bash
    python model_pathloss_data.py --fit-slit-dims-once
    ```

*   `--no-scatter`: Disable the scatter analysis and related plots.
    ```bash
    python model_pathloss_data.py --no-scatter
    ```

*   `--radius-bin-width <WIDTH>`: Set the bin width for smoothing the uncertainty vs. radius plot.
    ```bash
    python model_pathloss_data.py --radius-bin-width 0.02
    ```

## File Descriptions

*   `model_pathloss_data.py`: The main Python script for modeling, fitting, and plotting the NIRSpec pathloss data.
*   `jwst_nirspec_pathloss_erf.fits`: The output FITS file containing the modeled pathloss and variance arrays.
*   `jwst_nirspec_pathloss_erf_arrays_plot_by_wavelength.png`: A plot showing the pathloss and variance arrays from the FITS file for each wavelength.
*   `../data/flip_xy_slitlosses/`: The directory assumed to contain the input slit-loss data files (not included in this directory).

## Model Description

The core of this work is a physical model that describes the throughput of a Gaussian PSF through a rectangular slit. The throughput is calculated as the integral of the 2D Gaussian function over the area of the slit. This integral can be separated into two independent 1D integrals, which can be solved analytically using the error function (`erf`).

The model has four primary parameters that are determined by fitting to the data:

*   `slit_w`: The width of the slit.
*   `slit_h`: The height of the slit.
*   `psf_sx`: The standard deviation (sigma) of the Gaussian PSF in the x-direction.
*   `psf_sy`: The standard deviation (sigma) of the Gaussian PSF in the y-direction.

The script fits these parameters for each wavelength present in the input data.

## Outputs

The script generates two main types of outputs:

### Plots

A number of plots are generated to visualize the results of the analysis. These are saved in a new directory created each time the script is run (e.g., `plots_throughput_model_sparse1`).

*   **Throughput Fit Plots:** For each wavelength, a multi-panel plot showing the input data, the fitted model, the residuals (data - model), and various uncertainty plots.
*   **Parameter vs. Wavelength Plot:** A summary plot showing how the fitted model parameters (`slit_w`, `slit_h`, `psf_sx`, `psf_sy`) vary with wavelength.
*   **Scatter Summary Plot:** A summary plot showing the RMS scatter of the residuals as a function of the model pathloss for all wavelengths.
*   **FITS Array Plots:** If the `--plot-arrays` option is used, a plot showing the contents of the `PS` and `PSVAR` extensions of the input FITS file.

### FITS Files

*   `jwst_nirspec_pathloss_erf.fits`: This file is the primary data product. It is a multi-extension FITS file that contains the modeled pathloss (`PS` extension) and the corresponding variance (`PSVAR` extension) on a regular grid for a range of wavelengths. This file is created when using the `--update-fits` option.
*   `pathloss_prediction.fits` (or other specified name): A single-extension FITS file containing the predicted pathloss on a grid for a single wavelength, created when using the `--predict-wavelength` option.

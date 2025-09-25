# Physical model for PSF throughput through a rectangular slit.
# Model is based on the error function (erf) derived from integrating a Gaussian PSF.
# More robust initial guesses using percentiles
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from astropy.io import fits
import os
from typing import List, Tuple, Optional, Dict, Any
import glob
import argparse

# Fitting tools
try:
    from scipy.optimize import least_squares
    from scipy.special import erf
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    print("scipy and scipy.special not available: install scipy to enable fitting (pip install scipy).")

# -----------------------
# Load pathloss data measurements
# -----------------------
def parse_header_for_xy(header_line: str) -> Tuple[float, float]:
    """
    Parse the first header line to extract x and y offsets.

    Expected format example:
    "# wavelength (micron),  correction  , error correction; -0.012;-0.0848;"
    """
    parts = header_line.split(";")
    # Header ends with ";", so last split element may be empty
    if len(parts) < 3:
        raise ValueError(f"Unexpected header format: {header_line!r}")
    try:
        x_offset = float(parts[1].strip())
        y_offset = float(parts[2].strip())
    except Exception as exc:
        raise ValueError(f"Failed parsing x,y from header: {header_line!r}") from exc
    return x_offset, y_offset


def load_all_pathloss(
    data_dir: str = '../data/flip_xy_slitlosses',
    *,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all slit-loss text files into arrays for notebook use.

    Returns
    -------
    x : (N,) float
        X offsets per file.
    y : (N,) float
        Y offsets per file.
    wavelengths : (M,) float
        Wavelength grid (assumed identical across files).
    corrections : (N, M) float
        Correction ratios per file and wavelength.
    errors : (N, M) float
        Error on correction per file and wavelength.
    """
    pattern = os.path.join(data_dir, "slit_loss_correction_*.txt")
    file_paths = sorted(glob.glob(pattern))
    if not file_paths:
        raise FileNotFoundError(f"No files found matching {pattern}")

    x_list: List[float] = []
    y_list: List[float] = []
    corr_list: List[np.ndarray] = []
    err_list: List[np.ndarray] = []
    wavelengths_ref: Optional[np.ndarray] = None

    for fp in file_paths:
        try:
            with open(fp, "r") as fh:
                header = fh.readline().strip()
            x_off, y_off = parse_header_for_xy(header)
            arr = np.loadtxt(fp)
            if arr.ndim != 2 or arr.shape[1] < 3:
                if verbose:
                    print(f"Skipping malformed: {fp}")
                continue
            lam = arr[:, 0].astype(float)
            corr = arr[:, 1].astype(float)
            err = arr[:, 2].astype(float)

            if wavelengths_ref is None:
                wavelengths_ref = lam
            else:
                if lam.shape != wavelengths_ref.shape or not np.allclose(lam, wavelengths_ref, rtol=0, atol=1e-9):
                    raise ValueError("Wavelength grids differ between files; simplest loader assumes identical grids.")

            x_list.append(x_off)
            y_list.append(y_off)
            corr_list.append(corr)
            err_list.append(err)
        except Exception as exc:
            if verbose:
                print(f"Warning: failed to load {fp}: {exc}")
            continue

    if wavelengths_ref is None or not x_list:
        raise RuntimeError("No valid entries loaded.")

    x = np.asarray(x_list, dtype=float)
    y = np.asarray(y_list, dtype=float)
    corrections = np.vstack(corr_list)
    errors = np.vstack(err_list)
    
    # Clip data to [0, 1] range
    np.clip(corrections, 0, 1, out=corrections)

    return x, y, wavelengths_ref, corrections, errors

# -----------------------
# Physical (erf) Model
# -----------------------
def erf_profile(x, half_width, sigma):
    """Calculates the throughput of a 1D Gaussian PSF through a slit."""
    sigma_safe = np.where(sigma == 0, 1e-6, sigma)
    z1 = (half_width - x) / (np.sqrt(2) * sigma_safe)
    z2 = (-half_width - x) / (np.sqrt(2) * sigma_safe)
    return 0.5 * (erf(z1) - erf(z2))

def model(points, params):
    """Takes a coordinate array (N, 2) and parameters vector as input."""
    x = points[:, 0]
    y = points[:, 1]
    slit_w = params[0]
    slit_h = params[1]
    psf_sx = params[2]
    psf_sy = params[3]
    return evaluate_throughput_model(x, y, slit_w, slit_h, psf_sx, psf_sy)

def evaluate_throughput_model(x, y, slit_w, slit_h, psf_sx, psf_sy, xc=0, yc=0):
    """Evaluates the 2D throughput model at sparse (x, y) coordinates."""
    x_shifted = x - xc
    y_shifted = y - yc

    throughput_x = erf_profile(x_shifted, slit_w / 2.0, psf_sx)
    throughput_y = erf_profile(y_shifted, slit_h / 2.0, psf_sy)

    throughput_2d = throughput_x * throughput_y
    return throughput_2d

def evaluate_throughput_model_grid(ny, nx, grid_x_min, grid_x_max, grid_y_min, grid_y_max, slit_w, slit_h, psf_sx, psf_sy, xc=0, yc=0):
    """Evaluates the 2D throughput model on a grid."""
    y_grid, x_grid = np.mgrid[grid_y_min:grid_y_max:ny*1j, grid_x_min:grid_x_max:nx*1j]
    return evaluate_throughput_model(x_grid, y_grid, slit_w, slit_h, psf_sx, psf_sy, xc, yc)

# -----------------------
# Residuals & Fitting
# -----------------------
def residuals_erf_sparse(params, x, y, corrections):
    """Residuals function for the erf-based model with sparse data."""
    model = evaluate_throughput_model(x, y, *params)
    return (model - corrections).ravel()

def fit_throughput_model_sparse(x, y, corrections, initial_params, fixed_params=None):
    """Fits the physical throughput model to sparse data."""
    if not HAS_SCIPY:
        return {'success': False, 'message': 'scipy not installed'}

    # More robust initial guesses using percentiles
    x_p5, x_p95 = np.percentile(x, [5, 95])
    y_p5, y_p95 = np.percentile(y, [5, 95])
    slit_w_guess = (x_p95 - x_p5)
    slit_h_guess = (y_p95 - y_p5)

    if fixed_params and 'slit_w' in fixed_params and 'slit_h' in fixed_params:
        # Case 1: Slit dimensions are fixed, fit only for PSF sigmas.
        slit_w = fixed_params['slit_w']
        slit_h = fixed_params['slit_h']

        def residuals_fixed_slit(fit_params, x, y, corrections):
            # fit_params are [psf_sx, psf_sy]
            model = evaluate_throughput_model(x, y, slit_w, slit_h, fit_params[0], fit_params[1])
            return (model - corrections).ravel()

        param_keys = ['psf_sx', 'psf_sy']
        p0 = [
            initial_params.get('psf_sx', slit_w_guess / 4),
            initial_params.get('psf_sy', slit_h_guess / 4),
        ]
        lb = [0.05, 0.05]
        ub = [slit_w_guess, slit_h_guess]
        
        res = least_squares(residuals_fixed_slit, p0, args=(x, y, corrections), bounds=(lb, ub), verbose=0, xtol=1e-8, ftol=1e-8)
        
        params = {'slit_w': slit_w, 'slit_h': slit_h}
        params.update(dict(zip(param_keys, res.x)))

    else:
        # Case 2: Fit all four parameters.
        param_keys = ['slit_w', 'slit_h', 'psf_sx', 'psf_sy']
        p0 = [
            initial_params.get('slit_w', slit_w_guess),
            initial_params.get('slit_h', slit_h_guess),
            initial_params.get('psf_sx', slit_w_guess / 4),
            initial_params.get('psf_sy', slit_h_guess / 4),
        ]
        lb = [0.1, 0.1, 0.05, 0.05]
        ub = [slit_w_guess * 3, slit_h_guess * 3, slit_w_guess, slit_h_guess]
        
        res = least_squares(residuals_erf_sparse, p0, args=(x, y, corrections), bounds=(lb, ub), verbose=0, xtol=1e-8, ftol=1e-8)
        params = dict(zip(param_keys, res.x))

    params['xc'] = 0
    params['yc'] = 0
    return {'success': res.success, 'message': res.message, 'params': params}

# -----------------------
# Plotting & Main
# -----------------------
def calculate_2d_scatter(x, y, residuals, model_pathloss, nx=20, ny=20):
    """
    Calculates 2D scatter and bias by binning data points into a regular grid.
    
    Returns
    -------
    xi : 2D array
        X coordinates of bin centers
    yi : 2D array
        Y coordinates of bin centers
    total_uncertainty : 2D array
        Combined uncertainty (scatter and bias) in each bin
    scatter : 2D array
        RMS scatter in each bin
    bias : 2D array
        Mean bias in each bin
    bin_pathloss : 2D array
        Mean pathloss in each bin
    bin_counts : 2D array
        Number of points in each bin
    """
    # Create bin edges
    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)
    
    # Create bin centers for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xi, yi = np.meshgrid(x_centers, y_centers)
    
    # Initialize arrays for results
    scatter = np.full((ny, nx), np.nan)
    bias = np.full((ny, nx), np.nan)
    total_uncertainty = np.full((ny, nx), np.nan)
    bin_pathloss = np.full((ny, nx), np.nan)
    bin_counts = np.zeros((ny, nx), dtype=int)
    
    # Bin the data points
    x_indices = np.digitize(x, x_edges) - 1
    y_indices = np.digitize(y, y_edges) - 1
    
    # Calculate statistics for each bin
    for i in range(ny):
        for j in range(nx):
            mask = (x_indices == j) & (y_indices == i)
            points_in_bin = np.sum(mask)
            
            if points_in_bin > 1:  # Need at least 2 points for statistics
                bin_residuals = residuals[mask]
                # Calculate RMS scatter around the mean
                scatter[i, j] = np.sqrt(np.mean((bin_residuals - np.mean(bin_residuals))**2))
                # Calculate mean bias (systematic offset)
                bias[i, j] = np.mean(bin_residuals)
                # Combine scatter and bias in quadrature for total uncertainty
                total_uncertainty[i, j] = np.sqrt(scatter[i, j]**2 + bias[i, j]**2)
                bin_pathloss[i, j] = np.mean(model_pathloss[mask])
                bin_counts[i, j] = points_in_bin
    
    return xi, yi, total_uncertainty, scatter, bias, bin_pathloss, bin_counts

def model_uncertainty_from_pathloss(model_pathloss, data_pathloss, residuals, smooth_factor=0.5, spatial_smooth=1.0):
    """
    Creates a smooth 2D uncertainty model based on pathloss contours.
    Uses local averaging within similar pathloss values to create
    a continuous uncertainty field, with enforced monotonicity.
    
    Parameters
    ----------
    model_pathloss : 2D array
        The model pathloss values on a grid
    data_pathloss : 1D array
        The model pathloss values at data points
    residuals : 1D array
        The residuals at each data point
    smooth_factor : float
        Controls the width of the pathloss range used for local averaging
    spatial_smooth : float
        Controls the spatial smoothing of the final uncertainty map
        
    Returns
    -------
    uncertainty_model : 2D array
        Smoothed uncertainty values following pathloss contours
    raw_uncertainty : 2D array
        Unsmoothed uncertainty values for comparison
    """
    # Calculate RMS of residuals
    rms = np.sqrt(np.mean(residuals**2))
    
    # Create output array
    raw_uncertainty = np.zeros_like(model_pathloss)
    
    # For each point in the model grid
    for i in range(model_pathloss.shape[0]):
        for j in range(model_pathloss.shape[1]):
            # Get the pathloss value at this point
            pl = model_pathloss[i, j]
            
            # Find data points with similar pathloss values
            # Width of pathloss range scales with smooth_factor
            pl_mask = np.abs(data_pathloss - pl) < (smooth_factor * rms)
            
            if np.any(pl_mask):
                # Calculate local RMS for points with similar pathloss
                local_rms = np.sqrt(np.mean(residuals[pl_mask]**2))
                raw_uncertainty[i, j] = local_rms
            else:
                # If no similar points found, use global RMS
                raw_uncertainty[i, j] = rms
    
    # Apply spatial smoothing
    smoothed = gaussian_filter(raw_uncertainty, sigma=spatial_smooth)
    
    # Enforce monotonicity: uncertainty should generally increase as we move away from peak throughput
    peak_throughput = model_pathloss.max()
    dist_from_peak = np.abs(model_pathloss - peak_throughput)
    
    # Create a mask for the monotonic enforcement
    # We'll increase uncertainty based on distance from peak throughput
    monotonic_factor = 1.0 + 0.5 * (dist_from_peak / dist_from_peak.max())
    
    # Combine smoothed uncertainty with monotonic factor
    uncertainty_model = smoothed * monotonic_factor
    
    # Normalize to maintain similar overall scale
    uncertainty_model *= (raw_uncertainty.mean() / uncertainty_model.mean())
    
    return uncertainty_model, raw_uncertainty

def calculate_monotonic_uncertainty_2d(x_data, y_data, corrections_data, res_params, wavelength, radius_bin_width, target_x_grid, target_y_grid):
    """
    Calculates the 2D monotonic uncertainty map for a given model and data,
    interpolated onto a target grid.

    Parameters
    ----------
    x_data, y_data : 1D arrays
        Original x and y data points.
    corrections_data : 1D array
        Original correction values at data points.
    res_params : dict
        Dictionary of fitted model parameters (p from res).
    wavelength : float
        Wavelength for the current data.
    radius_bin_width : float
        The bin width for smoothing uncertainty vs. radius.
    target_x_grid : 2D array
        The x-coordinates of the target grid (e.g., FITS grid).
    target_y_grid : 2D array
        The y-coordinates of the target grid (e.g., FITS grid).

    Returns
    -------
    monotonic_uncertainty_2d_target_grid : 2D array
        The 2D monotonic uncertainty map interpolated onto the target grid.
    """
    p = res_params

    # Define a high-resolution grid for initial uncertainty model calculation
    x_range = p['slit_w'] * 1.5
    y_range = p['slit_h'] * 1.5
    grid_x_min, grid_x_max = -x_range, x_range
    grid_y_min, grid_y_max = -y_range, y_range
    nx_hi, ny_hi = 100, 100 # High resolution for initial calculation

    # Generate model on high-resolution grid
    model_hi = evaluate_throughput_model_grid(ny_hi, nx_hi, grid_x_min, grid_x_max, grid_y_min, grid_y_max,
                                              p['slit_w'], p['slit_h'], p['psf_sx'], p['psf_sy'], p['xc'], p['yc'])

    # Calculate residuals for uncertainty model
    model_at_data_points = evaluate_throughput_model(x_data, y_data, **p)
    residuals = corrections_data - model_at_data_points

    # Calculate uncertainty model
    z_data = np.column_stack((x_data, y_data))
    params_array = [p['slit_w'], p['slit_h'], p['psf_sx'], p['psf_sy']]
    data_pathloss = model(z_data, params_array)
    uncertainty_model, raw_uncertainty = model_uncertainty_from_pathloss(
        model_hi, data_pathloss, residuals, smooth_factor=0.5, spatial_smooth=1.0
    )

    # Calculate radius and flatten
    radius = np.sqrt(-2 * np.log(np.clip(model_hi, 1e-9, 1)))
    radius_flat = radius.flatten()
    uncertainty_flat = uncertainty_model.flatten()

    # Smooth uncertainty vs. radius
    smoothed_radius, smoothed_uncertainty = smooth_uncertainty_vs_radius(
        radius_flat, uncertainty_flat, radius_bin_width=radius_bin_width
    )

    # Calculate monotonic cap
    monotonic_uncertainty = np.maximum.accumulate(smoothed_uncertainty)

    # Evaluate model on the target grid to get target_radius
    target_model_image = evaluate_throughput_model(target_x_grid, target_y_grid, **p)
    target_radius = np.sqrt(-2 * np.log(np.clip(target_model_image, 1e-9, 1)))

    # Interpolate monotonic uncertainty onto the target grid
    monotonic_uncertainty_2d_target_grid = np.interp(target_radius.flatten(), smoothed_radius, monotonic_uncertainty).reshape(target_x_grid.shape)

    return monotonic_uncertainty_2d_target_grid

def smooth_uncertainty_vs_radius(radius_flat, uncertainty_flat, radius_bin_width=0.01):
    """
    Smooths uncertainty data by binning it based on radius.

    Parameters
    ----------
    radius_flat : np.ndarray
        Flattened array of radius values.
    uncertainty_flat : np.ndarray
        Flattened array of uncertainty values.
    radius_bin_width : float
        The width of each radius bin.

    Returns
    -------
    binned_radius : np.ndarray
        Centers of the radius bins.
    smoothed_uncertainty : np.ndarray
        Mean uncertainty in each radius bin.
    """
    # Sort by radius
    sort_idx = np.argsort(radius_flat)
    radius_sorted = radius_flat[sort_idx]
    uncertainty_sorted = uncertainty_flat[sort_idx]

    # Define bins
    min_radius = radius_sorted.min()
    max_radius = radius_sorted.max()
    bins = np.arange(min_radius, max_radius + radius_bin_width, radius_bin_width)

    # Digitize data into bins
    bin_indices = np.digitize(radius_sorted, bins)

    binned_radius = []
    smoothed_uncertainty = []

    # Calculate mean uncertainty for each bin
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        if np.any(mask):
            binned_radius.append(np.mean(radius_sorted[mask]))
            smoothed_uncertainty.append(np.mean(uncertainty_sorted[mask]))

    return np.array(binned_radius), np.array(smoothed_uncertainty)

def plot_results_sparse(x, y, corrections, res, wavelength, outdir, do_scatter=True, cmap='rainbow', radius_bin_width=0.01):
    """Plots the results for the sparse data fitting."""
    if do_scatter:
        fig = plt.figure(figsize=(24, 12))
        
        # Create a layout with 2 rows and 4 columns
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 4, figure=fig)
        
        # Create all subplot axes
        axes = np.array([fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]).reshape(2, 4)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = np.atleast_2d(axes)  # Ensure 2D array for consistency
    p = res['params']
    
    # Define grid for model visualization based on fit
    x_range = p['slit_w'] * 1.5
    y_range = p['slit_h'] * 1.5
    grid_x_min, grid_x_max = -x_range, x_range
    grid_y_min, grid_y_max = -y_range, y_range
    
    nx, ny = 100, 100
    
    model_image = evaluate_throughput_model_grid(ny, nx, grid_x_min, grid_x_max, grid_y_min, grid_y_max, 
                                                 p['slit_w'], p['slit_h'], p['psf_sx'], p['psf_sy'], p['xc'], p['yc'])
    model_at_data_points = evaluate_throughput_model(x, y, p['slit_w'], p['slit_h'], p['psf_sx'], p['psf_sy'], p['xc'], p['yc'])
    residuals = corrections - model_at_data_points

    # Plot 1: Data
    ax = axes[0, 0]
    im_data = ax.scatter(x, y, c=corrections, cmap=cmap, vmin=0, vmax=1)
    ax.set_title('Data')
    ax.set_aspect('equal', 'box')
    fig.colorbar(im_data, ax=ax, shrink=0.8)

    # Plot 2: Data over Model
    ax = axes[0, 1]
    im_model_bg = ax.imshow(model_image, origin='lower', 
                            extent=(grid_x_min, grid_x_max, grid_y_min, grid_y_max), 
                            cmap=cmap, vmin=0, vmax=1)
    ax.scatter(x, y, c=corrections, edgecolor='white', s=50, cmap=cmap, vmin=0, vmax=1)
    ax.set_title('Data over Model')
    ax.set_aspect('equal', 'box')
    fig.colorbar(im_model_bg, ax=ax, shrink=0.8)

    # Plot 3: Model
    ax = axes[0, 2]
    im_model = ax.imshow(model_image, origin='lower', 
                         extent=(grid_x_min, grid_x_max, grid_y_min, grid_y_max), 
                         cmap=cmap, vmin=0, vmax=1)
    ax.set_title('Fitted Model')
    ax.set_aspect('equal', 'box')
    fig.colorbar(im_model, ax=ax, shrink=0.8)

    # Plot 4: Residuals
    ax = axes[0, 3]
    im_res = ax.scatter(x, y, c=residuals, cmap='bwr', vmin=-0.2, vmax=0.2)
    ax.set_title('Residuals (Data $-$ Model)')
    ax.set_aspect('equal', 'box')
    fig.colorbar(im_res, ax=ax, shrink=0.8)

    if do_scatter:
        # Create high-resolution grid for smooth model
        nx_hi, ny_hi = 100, 100
        x_hi = np.linspace(x.min(), x.max(), nx_hi)
        y_hi = np.linspace(y.min(), y.max(), ny_hi)
        xi_hi, yi_hi = np.meshgrid(x_hi, y_hi)
        
        # Calculate model on high-res grid
        model_hi = evaluate_throughput_model(xi_hi, yi_hi, 
                                           p['slit_w'], p['slit_h'],
                                           p['psf_sx'], p['psf_sy'])
        
        # Calculate uncertainty model based on pathloss contours
        z = np.column_stack((x, y))
        params_array = [p['slit_w'], p['slit_h'], p['psf_sx'], p['psf_sy']]
        data_pathloss = model(z, params_array)
        uncertainty_model, raw_uncertainty = model_uncertainty_from_pathloss(model_hi, data_pathloss, residuals, 
                                                                          smooth_factor=0.5, spatial_smooth=1.0)

        # Calculate 2D binned statistics for scatter plot
        xi, yi, total_uncertainty, scatter_grid, bias_grid, bin_pathloss, bin_counts = calculate_2d_scatter(
            x, y, residuals, model_at_data_points, nx=15, ny=15)
        
        # Plot 5: 2D Total Uncertainty Map
        ax = axes[1, 0]
        mask = ~np.isnan(total_uncertainty)
        if np.any(mask):
            im_scatter = ax.imshow(total_uncertainty, origin='lower', 
                                extent=(x.min(), x.max(), y.min(), y.max()),
                                cmap='viridis', vmin=0, vmax=0.15)
            fig.colorbar(im_scatter, ax=ax, shrink=0.8, label='Total Uncertainty')
            
            for i in range(bin_counts.shape[0]):
                for j in range(bin_counts.shape[1]):
                    if bin_counts[i,j] > 0:
                        ax.text(xi[i,j], yi[i,j], str(bin_counts[i,j]),
                               ha='center', va='center', color='white',
                               fontsize=8, fontweight='bold')
        
        ax.set_title('2D Total Uncertainty Map (numbers show points per bin)')
        ax.set_aspect('equal', 'box')
        
        # Plot 6: Raw Uncertainty Model
        ax = axes[1, 1]
        im = ax.imshow(raw_uncertainty, origin='lower',
                      extent=(x.min(), x.max(), y.min(), y.max()),
                      cmap='viridis', vmin=0, vmax=0.15)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Raw Uncertainty')
        
        levels = np.linspace(0, 1, 11)
        ax.contour(xi_hi, yi_hi, model_hi, levels=levels,
                       colors='white', alpha=1, linewidths=0.5)
        
        ax.set_title('Raw Uncertainty Model (white lines show pathloss contours)')
        ax.set_aspect('equal', 'box')

        # --- New Plot Calculations for PS grid ---
        x_grid_ps, y_grid_ps = define_pathloss_grid()
        monotonic_uncertainty_on_ps_grid = calculate_monotonic_uncertainty_2d(
            x, y, corrections, p, wavelength, radius_bin_width, x_grid_ps, y_grid_ps
        )
        model_on_ps_grid = evaluate_throughput_model(x_grid_ps, y_grid_ps, **p)

        # Plot 7: Capped Uncertainty Model on PS Grid
        ax = axes[1, 2]
        im = ax.imshow(monotonic_uncertainty_on_ps_grid, origin='lower',
                      extent=(x_grid_ps.min(), x_grid_ps.max(), y_grid_ps.min(), y_grid_ps.max()),
                      cmap='viridis', vmin=0, vmax=0.15)
        fig.colorbar(im, ax=ax, shrink=0.8, label='Capped Uncertainty on PS Grid')
        
        ax.contour(x_grid_ps, y_grid_ps, model_on_ps_grid, levels=levels,
                       colors='white', alpha=1, linewidths=0.5)
        
        ax.set_title('Capped Monotonic Uncertainty on PS Grid')
        ax.set_aspect('equal', 'box')

        # Recalculate for Plot 8 (Broad Uncertainty Model vs. Radius)
        radius = np.sqrt(-2 * np.log(np.clip(model_hi, 1e-9, 1)))
        radius_flat = radius.flatten()
        uncertainty_flat = uncertainty_model.flatten()
        smoothed_radius, smoothed_uncertainty = smooth_uncertainty_vs_radius(
            radius_flat, uncertainty_flat, radius_bin_width=radius_bin_width
        )
        monotonic_uncertainty = np.maximum.accumulate(smoothed_uncertainty)

        # Plot 8: Broad Uncertainty Model vs. Radius (with monotonic cap)
        ax = axes[1, 3]
        # Plot original uncertainty_sorted for comparison
        sort_idx = np.argsort(radius_flat)
        radius_sorted_original = radius_flat[sort_idx]
        uncertainty_sorted_original = uncertainty_flat[sort_idx]
        ax.plot(radius_sorted_original, uncertainty_sorted_original, 'b-', alpha=0.5, label='Uncertainty Model (Raw)')
        ax.plot(smoothed_radius, smoothed_uncertainty, 'g-', alpha=0.7, label='Uncertainty Model (Smoothed)')
        ax.plot(smoothed_radius, monotonic_uncertainty, 'r-', linewidth=2, label='Monotonic Cap')

        ax.set_xlabel("Radius (from model contours)")
        ax.set_ylabel("Uncertainty")
        ax.set_title("Broad Uncertainty Model vs. Radius")
        ax.grid(True)
        ax.set_ylim(bottom=0, top=0.2)
        ax.legend()

    else:
        pass

    # Adjust labels for all 2D plots
    for ax in axes.flat:
        # Check if it has an xlabel, which indicates it's one of the main plots
        if ax.get_xlabel() or ax.get_ylabel():
             continue # Skip plots that already have labels (like the radius plot)
        ax.set_xlabel("x offset")
        ax.set_ylabel("y offset")
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)

    title = f"{wavelength:.2f} µm"
    title += f"\nSlit: {p['slit_w']:.3f}x{p['slit_h']:.3f}"
    title += f"\nPSF σ: ({p['psf_sx']:.3f}, {p['psf_sy']:.3f})"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        filename = f"{outdir}/throughput_fit_sparse_{wavelength:.2f}um.png"
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to {filename}")
        plt.close(fig)

def plot_scatter_vs_pathloss(pathloss_bins, rms_scatter, wavelength, outdir):
    """Plots the RMS scatter vs. pathloss."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pathloss_bins, rms_scatter, 'o-', label=f'{wavelength:.2f} µm')
    ax.set_xlabel("Model Pathloss")
    ax.set_ylabel("RMS Scatter (Data - Model)")
    ax.set_title(f"RMS Scatter vs. Pathloss at {wavelength:.2f} µm")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if outdir:
        filename = f"{outdir}/scatter_vs_pathloss_{wavelength:.2f}um.png"
        plt.savefig(filename, dpi=150)
        print(f"Scatter plot saved to {filename}")
        plt.close(fig)

def plot_params_vs_wavelength(all_results, outdir):
    """Plots the fitted model parameters vs. wavelength on a single plot."""
    
    wavelengths = [res['wavelength'] for res in all_results]
    param_keys = ['slit_w', 'slit_h', 'psf_sx', 'psf_sy']
    params = {key: [res['params'][key] for res in all_results] for key in param_keys}

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for key in param_keys:
        ax.plot(wavelengths, params[key], 'o-', label=key)

    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Model Parameters vs. Wavelength", fontsize=16)
    ax.grid(True)
    ax.legend()
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        filename = f"{outdir}/model_params_vs_wavelength.png"
        plt.savefig(filename, dpi=150)
        print(f"Summary plot saved to {filename}")
        plt.close(fig)

def plot_scatter_summary(all_results, outdir):
    """Plots the RMS scatter vs. pathloss for all wavelengths on a single plot."""
    # Make figure with custom size to accommodate legend and colorbar
    fig = plt.figure(figsize=(12, 8))
    
    # Create gridspec with space for plot, colorbar, and legend
    # Ratios: [plot] [small gap] [colorbar] [legend]
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 20, figure=fig)
    ax = fig.add_subplot(gs[0:15])  # Main plot
    cax = fig.add_subplot(gs[16:17])  # Colorbar
    # Legend will be placed to the right of the colorbar
    
    # Sort results by wavelength for consistent color mapping
    sorted_results = sorted(all_results, key=lambda x: x['wavelength'])
    wavelengths = np.array([res['wavelength'] for res in sorted_results])
    
    # Create color map that maps wavelengths to colors
    norm = Normalize(wavelengths.min(), wavelengths.max())
    cmap = plt.cm.get_cmap('rainbow')
    
    for result in sorted_results:
        if 'scatter_bins' in result and 'scatter_rms' in result:
            color = cmap(norm(result['wavelength']))
            ax.plot(result['scatter_bins'], result['scatter_rms'], 'o-', 
                   color=color, label=f"{result['wavelength']:.2f} µm")

    ax.set_xlabel("Model Pathloss")
    ax.set_ylabel("RMS Scatter (Data - Model)")
    ax.set_title("RMS Scatter vs. Pathloss", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.2)
    ax.grid(True)
    
    # Add colorbar to show wavelength scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Wavelength (µm)')
    
    # Place legend to the right of the colorbar
    legend = ax.legend(bbox_to_anchor=(1.45, 1), loc='upper left',
                      borderaxespad=0, ncol=1)
    
    # Adjust layout but leave room for legend
    plt.tight_layout(rect=(0, 0, 0.85, 1))

    if outdir:
        filename = f"{outdir}/scatter_summary.png"
        plt.savefig(filename, dpi=150)
        print(f"Scatter summary plot saved to {filename}")
        plt.close(fig)

def plot_pathloss_arrays_from_fits(filepath: str, outdir: str = ".", var_max=0.01):  # 0.02
    """
    Plots all the PS and PSVAR arrays from a pathloss FITS file.
    Creates a large plot with one column per wavelength and consistent styling.
    """
    try:
        with fits.open(filepath) as hdul:
            # Get wavelength information
            try:
                ps_hdu_for_wave = hdul[('PS', 1)]
                wcs_header = ps_hdu_for_wave.header
                nwave = wcs_header['NAXIS3']
                crpix3 = wcs_header.get('CRPIX3', 1)
                crval3 = wcs_header.get('CRVAL3', 1)
                cdelt3 = wcs_header.get('CDELT3', 1)
                wavelengths = (crval3 + (np.arange(nwave) - (crpix3 - 1)) * cdelt3) * 1e6 # microns
            except (KeyError, IndexError):
                print("Could not determine wavelengths from FITS header.")
                return

            # Get and group extensions
            ps_hdus = [h for h in hdul if h.name == 'PS']
            psvar_hdus = [h for h in hdul if h.name == 'PSVAR']

            groups = [
                [h for h in ps_hdus if h.header.get('EXTVER', 1) != 2],
                [h for h in psvar_hdus if h.header.get('EXTVER', 1) != 2],
                [h for h in ps_hdus if h.header.get('EXTVER', 1) == 2],
                [h for h in psvar_hdus if h.header.get('EXTVER', 1) == 2]
            ]
            group_titles = ["PS", "PSVAR", "PS (1x3)", "PSVAR (1x3)"]

            active_groups_with_titles = [(g, t) for g, t in zip(groups, group_titles) if g]

            if not active_groups_with_titles:
                print(f"No 'PS' or 'PSVAR' extensions found in {filepath}")
                return

            nrows = len(active_groups_with_titles)
            ncols = nwave

            height_ratios = [3 if '(1x3)' in t else 1 for g, t in active_groups_with_titles]

            fig = plt.figure(figsize=(ncols, sum(height_ratios) * 1.5))
            gs = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.05, height_ratios=height_ratios)
            axes = gs.subplots(sharex='col')

            if nrows == 1 and ncols > 1: axes = axes.reshape(1, -1)
            if ncols == 1 and nrows > 1: axes = axes.reshape(-1, 1)
            if nrows == 1 and ncols == 1: axes = np.array([[axes]])


            for i, (group, title) in enumerate(active_groups_with_titles):
                hdu = group[0]
                is_var = 'VAR' in title
                is_1x3 = '(1x3)' in title
                vmin, vmax = (0, var_max) if is_var else (0, 1)

                for j in range(ncols):
                    ax = axes[i, j]
                    
                    data_slice = hdu.data[j, :, :]
                    if is_var:
                        cmap = 'viridis'
                    else:
                        cmap = 'rainbow'
                    im = ax.imshow(data_slice, origin='lower', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

                    # Titles only on top row
                    if i == 0:
                        ax.set_title(f"{wavelengths[j]:.2f}")
                        #ax.set_title(f"{wavelengths[j]:.2f} µm")

                    # Y-axis label only on leftmost plot
                    if j == 0:
                        ax.set_ylabel(title)
                    else:
                        ax.tick_params(labelleft=False)
                    
                    # Colorbar only on rightmost plot
                    if j == ncols - 1:
                        #cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                        #cbar.set_label("Variance" if is_var else "Pathloss")

                        from mpl_toolkits.axes_grid1 import make_axes_locatable

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="50%", pad=0.05)  # colorbar width and spacing
                        cbar = fig.colorbar(im, cax=cax)
                        cbar.set_label("Variance" if is_var else "Pathloss")

                    ny, nx = data_slice.shape
                    ax.set_xticks(np.arange(0, nx, 20))
                    ax.set_yticks(np.arange(0, ny, 10))

            fig.suptitle(f"Pathloss Arrays from {os.path.basename(filepath)}", fontsize=16, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 1], h_pad=0, w_pad=0)
            #plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=0.5, w_pad=0.5)
            
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                base_filename = os.path.splitext(os.path.basename(filepath))[0]
                filename = os.path.join(outdir, f"{base_filename}_arrays_plot_by_wavelength.png")
                plt.savefig(filename, dpi=150)
                print(f"Plot saved to {filename}")
                plt.close(fig)
            else:
                plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")


def model_and_plot_all_wavelengths_sparse(x, y, wavelengths, corrections, errors, outdir, skip_individual_plots=False, fit_slit_dims_once=False):
    """Main loop to model and plot sparse data for all wavelengths."""
    print("--- Starting Batch Processing (Sparse Throughput Model) ---")

    initial_params = {}
    all_results = []
    fixed_slit_params = None

    for i_wl, wavelength in enumerate(wavelengths):
        print(f"Fitting wavelength {i_wl + 1}/{len(wavelengths)}: {wavelength:.2f} µm")
        
        correction_slice = corrections[:, i_wl]
        
        current_fixed_params = None
        if fit_slit_dims_once and fixed_slit_params:
            current_fixed_params = fixed_slit_params

        res = fit_throughput_model_sparse(x, y, correction_slice, initial_params, fixed_params=current_fixed_params)
        
        if res['success']:
            if fit_slit_dims_once and i_wl == 0:
                fixed_slit_params = {
                    'slit_w': res['params']['slit_w'],
                    'slit_h': res['params']['slit_h'],
                }
            
            initial_params = res['params']
            
            # Measure and plot scatter
            model_at_data_points = evaluate_throughput_model(x, y, **res['params'])
            residuals = correction_slice - model_at_data_points
            pathloss_bins, rms_scatter = measure_scatter_vs_pathloss(model_at_data_points, residuals)

            if not skip_individual_plots:
                plot_results_sparse(x, y, correction_slice, res, wavelength, outdir)
                plot_scatter_vs_pathloss(pathloss_bins, rms_scatter, wavelength, outdir)
            
            result_data = {
                'wavelength': wavelength, 
                'params': res['params'],
                'scatter_bins': pathloss_bins,
                'scatter_rms': rms_scatter
            }
            all_results.append(result_data)
        else:
            print(f"  Fit failed for wavelength {wavelength:.2f} µm: {res['message']}")
            initial_params = {} # Reset on failure
            if fit_slit_dims_once and i_wl == 0:
                print("  Cannot proceed with fixed slit dimensions as the first fit failed.")
                return []

    if all_results:
        print("\n--- Generating Final Summary Plots ---")
        plot_params_vs_wavelength(all_results, outdir)
        plot_scatter_summary(all_results, outdir)
    
    return all_results


def interpolate_params(target_wavelength, all_results):
    """Interpolates model parameters to a specific wavelength."""
    if not all_results:
        raise ValueError("Cannot interpolate without model results. Run the analysis first.")

    wavelengths = np.array([res['wavelength'] for res in all_results])
    param_keys = ['slit_w', 'slit_h', 'psf_sx', 'psf_sy']
    
    # Sort by wavelength to ensure np.interp works correctly
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    #print(target_wavelength, wavelengths)
    
    interpolated_params = {}
    for key in param_keys:
        values = np.array([res['params'][key] for res in all_results])[sort_idx]
        interpolated_value = np.interp(target_wavelength, wavelengths, values)
        interpolated_params[key] = interpolated_value
        
    interpolated_params['xc'] = 0
    interpolated_params['yc'] = 0
        
    return interpolated_params

def define_pathloss_grid(extend_grid=False):
    nx = 31
    ny = 31  # will extend below if needed
    #ny = 31 if not extend_grid else 91
    
    x_coords = np.linspace(-0.5, 0.5, nx)
    y_coords = np.linspace(-0.5, 0.5, ny) 
    if extend_grid:
        y_coords = np.concatenate([y_coords[:-1], y_coords[:-1], y_coords])

    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    return x_grid, y_grid

def calculate_and_save_pathloss_grid(params, wavelength, out_filepath, extend_grid=False):
    """Calculates pathloss on a grid and saves it as a FITS file."""

    x_grid, y_grid = define_pathloss_grid(extend_grid=extend_grid)
    pathloss_grid = evaluate_throughput_model(x_grid, y_grid, **params)

    # Create FITS file
    hdu = fits.PrimaryHDU(pathloss_grid)
    
    # Add metadata to header
    hdr = hdu.header
    hdr['WAVELEN'] = (wavelength, 'Target wavelength (microns)')
    hdr['SLIT_W'] = (params['slit_w'], 'Interpolated slit width')
    hdr['SLIT_H'] = (params['slit_h'], 'Interpolated slit height')
    hdr['PSF_SX'] = (params['psf_sx'], 'Interpolated PSF sigma_x')
    hdr['PSF_SY'] = (params['psf_sy'], 'Interpolated PSF sigma_y')
    hdr['EXTEND'] = (extend_grid, 'Grid was extended to 91x31')

    hdu.writeto(out_filepath, overwrite=True)
    print(f"Pathloss grid saved to {out_filepath}")

def update_pathloss_fits(all_results, x, y, corrections, wavelengths, infile='jwst_nirspec_pathloss_0005.fits', outfile='jwst_nirspec_pathloss_erf.fits', radius_bin_width=0.01):
    """
    Updates a pathloss FITS file with model-generated data.
    """
    if os.path.exists(infile):
        print(f"\n--- Updating FITS file {infile} ---")
    else:
        print(f"Error: Input FITS file not found at {infile}")
        raise FileNotFoundError(f"Input FITS file not found: {infile}")

    with fits.open(infile) as hdu_list:
        hdu_list.info()
        data = hdu_list['PS', 1].data
        hdr  = hdu_list['PS', 1].header
        #nwave = data.shape[0]
        nwave, ny, nx = data.shape
        crpix3, crval3, cdelt3 = hdr['CRPIX3'], hdr['CRVAL3'], hdr['CDELT3']
        target_wavelengths = crval3 + (np.arange(nwave) + 1 - crpix3) * cdelt3
        target_wavelengths *= 1e6
        print(f"Target wavelengths: {target_wavelengths}")

        for extver in [1, 2]:
            print(f"Processing extension version {extver}...")
            ps_hdu = hdu_list[('PS', extver)]
            
            # Get spatial grid coordinates
            extend_grid = extver == 2  # Use extended grid for second extension
            x_grid, y_grid = define_pathloss_grid(extend_grid=extend_grid)

            # Generate new pathloss data cube
            new_ps_data = np.zeros_like(ps_hdu.data)
            new_psvar_data = np.zeros_like(ps_hdu.data)

            for i, wl in enumerate(target_wavelengths):
                params = interpolate_params(wl, all_results)
                new_ps_data[i, :, :] = evaluate_throughput_model(x_grid, y_grid, **params)

                # Find the index of the closest wavelength in the data to get corrections
                closest_wl_idx = np.argmin(np.abs(wavelengths - wl))
                correction_slice = corrections[:, closest_wl_idx]

                # Calculate monotonic uncertainty and square it for variance
                monotonic_uncertainty_2d = calculate_monotonic_uncertainty_2d(
                    x, y, correction_slice, params, wl, radius_bin_width, x_grid, y_grid
                )
                new_psvar_data[i, :, :] = monotonic_uncertainty_2d**2
            
            # Update PS and PSVAR data
            ps_hdu.data = new_ps_data
            ps_hdu.header['HISTORY'] = 'Data replaced with erf model values.'
            
            psvar_hdu = hdu_list[('PSVAR', extver)]
            psvar_hdu.data = new_psvar_data
            psvar_hdu.header['HISTORY'] = 'Data replaced with monotonic uncertainty model (squared).'

        print(f"Saving updated FITS file to {outfile}...")
        hdu_list.writeto(outfile, overwrite=True)
        print("FITS file update complete.")

def measure_scatter_vs_pathloss(model_pathloss, residuals, n_bins=10):
    """
    Measures the RMS scatter of residuals in bins of model pathloss.

    Parameters
    ----------
    model_pathloss : np.ndarray
        The pathloss values from the model for each data point.
    residuals : np.ndarray
        The data-model residuals for each data point.
    n_bins : int
        The number of bins to use for pathloss values.

    Returns
    -------
    pathloss_bin_centers : np.ndarray
        The center of each pathloss bin.
    rms_scatter : np.ndarray
        The RMS scatter in each bin.
    """
    # Define bins from 0 to 1, as pathloss is a throughput value
    pathloss_bins = np.linspace(0, 1, n_bins + 1)
    pathloss_bin_centers = (pathloss_bins[:-1] + pathloss_bins[1:]) / 2

    # Digitize the model pathloss values into the bins
    bin_indices = np.digitize(model_pathloss, pathloss_bins)

    rms_scatter = []
    # Calculate RMS for each bin
    for i in range(1, n_bins + 1):
        residuals_in_bin = residuals[bin_indices == i]
        if len(residuals_in_bin) > 1:  # Need at least 2 points to measure scatter
            rms = np.sqrt(np.mean(residuals_in_bin**2))
            rms_scatter.append(rms)
        else:
            rms_scatter.append(np.nan)

    return pathloss_bin_centers, np.array(rms_scatter)

# -----------------------
# Plotting & Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model pathloss data and generate plots.')
    parser.add_argument('--plot-arrays', type=str, help='Input FITS file to plot arrays from.')
    parser.add_argument('--summary-only', action='store_true', help='Only generate the final summary plot.')
    parser.add_argument('--predict-wavelength', type=float, help='Wavelength to predict pathloss for.')
    parser.add_argument('--extend-grid', action='store_true', help='Extend the prediction grid to 91x31.')
    parser.add_argument('--output-fits', type=str, default='pathloss_prediction.fits', help='Output FITS file for prediction.')
    parser.add_argument('--update-fits', action='store_true', help='Update jwst_nirspec_pathloss_0005.fits with the model.')
    parser.add_argument('--fit-slit-dims-once', action='store_true', help='Fit slit dimensions for the first wavelength only and keep them constant.')
    parser.add_argument('--no-scatter', action='store_true', help='Disable scatter analysis and related plots.')
    parser.add_argument('--radius-bin-width', type=float, default=0.01,
                        help='Bin width for smoothing uncertainty vs. radius plot (default: 0.01).')
    args = parser.parse_args()

    base_outdir = "plots_throughput_model_sparse"
    run_num = 1
    while os.path.exists(f"{base_outdir}{run_num}"):
        run_num += 1
    outdir = f"{base_outdir}{run_num}"

    if args.plot_arrays:
        print(f"--- Plotting arrays from {args.plot_arrays} ---")
        plot_pathloss_arrays_from_fits(args.plot_arrays, outdir=outdir)
        exit()

    print("Loading all sparse pathloss data...")
    x, y, wavelengths, corrections, errors = load_all_pathloss(verbose=True)
    
    # Always run the modeling to get parameters
    all_results = model_and_plot_all_wavelengths_sparse(x, y, wavelengths, corrections, errors, outdir, skip_individual_plots=True, fit_slit_dims_once=args.fit_slit_dims_once)

    if args.predict_wavelength:
        print("\n--- Running in Prediction Mode ---")
        print(f"Interpolating parameters for wavelength {args.predict_wavelength:.2f} µm...")
        interpolated_params = interpolate_params(args.predict_wavelength, all_results)
        calculate_and_save_pathloss_grid(interpolated_params, args.predict_wavelength, args.output_fits, args.extend_grid)
        print("\nPrediction complete.")
    elif args.update_fits:
        update_pathloss_fits(all_results, x, y, corrections, wavelengths, infile='jwst_nirspec_pathloss_0005.fits', outfile='jwst_nirspec_pathloss_erf.fits', radius_bin_width=args.radius_bin_width)
    else:
        print(f"\n--- Running in Plotting Mode ---")
        print(f"Output will be saved to '{outdir}'")
        # Re-run plotting functions now that parameter generation is complete
        if not args.summary_only:
            for result in all_results:
                i_wl = np.where(wavelengths == result['wavelength'])[0][0]
                correction_slice = corrections[:, i_wl]
                plot_results_sparse(x, y, correction_slice, result, result['wavelength'], outdir, do_scatter=not args.no_scatter, radius_bin_width=args.radius_bin_width)
        
        if all_results:
            print("\n--- Generating Final Summary Plot ---")
            plot_params_vs_wavelength(all_results, outdir)
        print("\nProcessing complete.")

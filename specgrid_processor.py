# specgrid_processor.py
# Author: Soroush Arabi
# All rights reserved

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter
from nanonispy import read
import logging
import csv

class SpecgridAnalyser:
    """
    A class to analyze and plot spectroscopy data from .3ds files.
    """

    def __init__(self, specgrid_path):
        self.specgrid_path = specgrid_path
        self.peaks = {}
        self.filtering_method = "Savitzky-Golay Filter"

        # State Tracking Attributes
        self.filter_applied = False
        self.linear_fit_subtracted = False

        # Configure logging to suppress INFO messages
        logging.basicConfig(level=logging.WARNING)

        try:
            self.grid_dic = read.Grid(self.specgrid_path)
        except Exception as e:
            logging.error(f"Error loading specgrid data: {e}")
            raise

        try:
            if 'LockIn (V)' in self.grid_dic.signals:
                self.lockin_v_raw = self.grid_dic.signals['LockIn (V)'].data
            elif 'Current (A)' in self.grid_dic.signals:
                self.lockin_v_raw = self.grid_dic.signals['Current (A)'].data
            else:
                logging.error("No spectroscopy data found in signals.")
                raise AttributeError("No spectroscopy data found in signals.")

            self.lockin_v_raw = np.array(self.lockin_v_raw)

            if self.lockin_v_raw.ndim == 4:
                if self.lockin_v_raw.shape[3] != 1:
                    logging.warning(f"Data points dimension is {self.lockin_v_raw.shape[3]}, expected 1. Proceeding by taking the first data point.")
                self.lockin_v_raw = self.lockin_v_raw[:, :, :, 0]
            elif self.lockin_v_raw.ndim == 3:
                self.lockin_v_raw = np.transpose(self.lockin_v_raw, (2, 0, 1))
            else:
                logging.error(f"Spectroscopy data must be either 3D or 4D array, but got shape {self.lockin_v_raw.shape}")
                raise ValueError(f"Spectroscopy data must be either 3D or 4D array, but got shape {self.lockin_v_raw.shape}")

            if 'sweep_signal' in self.grid_dic.signals:
                self.bias = np.array(self.grid_dic.signals['sweep_signal'].data)
            else:
                logging.error("No 'sweep_signal' found in signals.")
                raise AttributeError("No 'sweep_signal' found in signals.")

            if self.bias.ndim != 1:
                logging.error(f"Bias vector must be 1D, but got shape {self.bias.shape}")
                raise ValueError(f"Bias vector must be 1D, but got shape {self.bias.shape}")

            self.Y_pixel_size, self.X_pixel_size = self.lockin_v_raw.shape[1], self.lockin_v_raw.shape[2]
            self.x = np.arange(self.X_pixel_size + 1)
            self.y = np.arange(self.Y_pixel_size + 1)
            self.X_mesh, self.Y_mesh = np.meshgrid(self.x, self.y)

            self.lockin_v_filtered = self.lockin_v_raw.copy()

            # Initialize peak intensity grid
            self.peak_intensity_grid = np.zeros((self.Y_pixel_size, self.X_pixel_size))

            # Initialize attribute to store original filtered data for linear subtraction
            self.lockin_v_filtered_original = self.lockin_v_filtered.copy()

        except Exception as e:
            logging.error(f"Error processing grid data: {e}")
            raise

    def apply_sg_filter_and_detect_peaks(self, window_length=5, polyorder=2, min_distance=2, threshold=0.05):
        """
        Apply Savitzky-Golay filter to each pixel's spectrum and detect peaks.
        """
        logging.info("Applying Savitzky-Golay filter and detecting peaks for each pixel.")

        # Reset filtered data to raw data
        self.lockin_v_filtered = self.lockin_v_raw.copy()

        # Apply linear fit subtraction if enabled
        if self.linear_fit_subtracted:
            self.subtract_linear_fit()

        if window_length % 2 == 0:
            window_length += 1  # Ensure window length is odd
        if window_length > len(self.bias):
            window_length = len(self.bias) - 1
            if window_length % 2 == 0:
                window_length -= 1
            logging.warning(f"Adjusted window_length to {window_length} to fit data length.")

        self.peaks = {}
        total_peaks = 0
        peak_intensity_grid = np.zeros((self.Y_pixel_size, self.X_pixel_size))  # Store peak counts

        for y in range(self.Y_pixel_size):
            for x in range(self.X_pixel_size):
                y_data = self.lockin_v_filtered[:, y, x]  # Use adjusted data

                # Apply Savitzky-Golay filter to individual pixel data
                y_filtered = savgol_filter(y_data, window_length=window_length, polyorder=polyorder)

                # Store the filtered data
                self.lockin_v_filtered[:, y, x] = y_filtered

                # Detect peaks in the filtered spectrum
                peaks_indices, properties = find_peaks(
                    y_filtered,
                    height=threshold,
                    distance=min_distance
                )

                peak_list = []
                for peak_idx in peaks_indices:
                    peak_bias = self.bias[peak_idx]
                    peak_intensity = y_filtered[peak_idx]  # Use filtered data for intensity
                    peak_list.append((peak_bias, peak_intensity))

                # Store detected peaks
                if peak_list:
                    self.peaks[(x, y)] = peak_list
                    peak_intensity_grid[y, x] = len(peak_list)
                    total_peaks += len(peak_list)

        self.peak_intensity_grid = peak_intensity_grid
        logging.info(f"Completed peak detection. Total peaks detected across all pixels: {total_peaks}")

        # Update the original filtered data after peak detection
        self.lockin_v_filtered_original = self.lockin_v_filtered.copy()

        # Update filter_applied flag
        self.filter_applied = True

    def plotting_specgrid(self, bias_min, bias_max, apply_gaussian_filter=False, fig_width=5, fig_height=4, save_fig=False):
        """
        Plot the distribution of peaks across the pixel grid for a given bias window.

        Parameters:
        - bias_min (float): Minimum bias voltage (V).
        - bias_max (float): Maximum bias voltage (V).
        - apply_gaussian_filter (bool): Whether to apply Gaussian filtering to the peak counts.
        - fig_width (float): Width of the figure.
        - fig_height (float): Height of the figure.
        - save_fig (bool): Whether to save the plot as a PDF.

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure.
        """
        # Initialize a grid to store peak counts within the bias window
        peak_counts_in_window = np.zeros((self.Y_pixel_size, self.X_pixel_size))

        for y in range(self.Y_pixel_size):
            for x in range(self.X_pixel_size):
                count = 0
                if (x, y) in self.peaks:
                    for peak_bias, _ in self.peaks[(x, y)]:
                        if bias_min <= peak_bias <= bias_max:
                            count += 1
                peak_counts_in_window[y, x] = count

        if apply_gaussian_filter:
            peak_counts_in_window = gaussian_filter(peak_counts_in_window, sigma=1)

        # Store the current peak counts as an attribute for saving
        self.current_peak_counts = peak_counts_in_window.copy()

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        cmap = plt.cm.plasma
        c = ax.pcolormesh(
            self.X_mesh, self.Y_mesh, peak_counts_in_window,
            shading='flat', cmap=cmap
        )

        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Number of Peaks per Pixel', fontsize=14)

        ax.set_xlabel('$X$', fontsize=14)
        ax.set_ylabel('$Y$', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set number of ticks and format tick labels
        ax.set_xticks(np.linspace(0, self.X_pixel_size, 5))
        ax.set_yticks(np.linspace(0, self.Y_pixel_size, 5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}'))

        ax.set_xlim(0, self.X_pixel_size)
        ax.set_ylim(0, self.Y_pixel_size)

        # Remove grid and title
        ax.grid(False)

        plt.tight_layout()

        if save_fig:
            filename = 'grid_plot.pdf'
            try:
                plt.savefig(filename, transparent=True)
                logging.info(f"Specgrid plot saved as {filename}")
            except Exception as e:
                logging.error(f"Error saving specgrid plot: {e}")

        return fig  # Return figure instead of plt.show()

    def plotting_average_spectrum(self, bias_min, bias_max, fig_width=5, fig_height=4, save_fig=False):
        """
        Plot the average dI/dV spectrum of pixels with at least one peak within the specified bias window.

        Parameters:
        - bias_min (float): Minimum bias voltage (V).
        - bias_max (float): Maximum bias voltage (V).
        - fig_width (float): Width of the figure.
        - fig_height (float): Height of the figure.
        - save_fig (bool): Whether to save the plot as a PDF.

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure.
        """
        # Identify pixels with at least one peak in the bias window
        pixels_with_peaks_in_window = []

        for y in range(self.Y_pixel_size):
            for x in range(self.X_pixel_size):
                if (x, y) in self.peaks:
                    peaks_in_window = [(pb, pi) for pb, pi in self.peaks[(x, y)] if bias_min <= pb <= bias_max]
                    if peaks_in_window:
                        pixels_with_peaks_in_window.append((x, y))

        if not pixels_with_peaks_in_window:
            logging.warning("No pixels with detected peaks in the specified bias window.")
            average_spectrum = np.zeros_like(self.bias)
        else:
            spectra = []
            for x, y in pixels_with_peaks_in_window:
                spectra.append(self.lockin_v_filtered[:, y, x])  # Use filtered data
            average_spectrum = np.mean(spectra, axis=0)

        # Store the current average spectrum as an attribute for saving
        self.current_average_spectrum = average_spectrum.copy()

        # Plot the average spectrum
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.plot(self.bias * 1e3, average_spectrum, color='tab:blue')

        ax.set_xlabel('$V_b$ (mV)', fontsize=14)
        ax.set_ylabel('$\\overline{dI/dV}$ (arb. units)', fontsize=14)

        # Adjust ticks
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Set number of ticks and format tick labels
        ax.locator_params(axis='x', nbins=7)
        ax.locator_params(axis='y', nbins=5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.2f}'))

        # Highlight the bias window with more transparency
        ax.axvspan(bias_min * 1e3, bias_max * 1e3, color='tab:red', alpha=0.15)

        # Remove grid and title
        ax.grid(False)

        plt.tight_layout()

        if save_fig:
            filename = 'average_spectrum_plot.pdf'
            csv_filename = 'average_spectrum_data.csv'
            try:
                plt.savefig(filename, transparent=True)
                logging.info(f"Average spectrum plot saved as {filename}")
            except Exception as e:
                logging.error(f"Error saving average spectrum plot: {e}")

            # Save average spectrum data to CSV
            try:
                with open(csv_filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Bias_V', 'Average_dI_dV'])
                    for bias, avg in zip(self.bias, self.current_average_spectrum):
                        csvwriter.writerow([bias, avg])
                logging.info(f"Average spectrum data saved as {csv_filename}")
            except Exception as e:
                logging.error(f"Error saving average spectrum data: {e}")

        return fig  # Return figure instead of plt.show()

    def plot_peak_histogram(self, bias_min=None, bias_max=None, fig_width=5, fig_height=4, save_fig=False):
        """
        Plot a histogram of the number of peaks found at different biases within an optional bias window.
        If bias_min and bias_max are provided, only peaks within this range are considered.
        """
        # Collect all peak biases
        peak_biases = []
        for peak_list in self.peaks.values():
            for peak_bias, _ in peak_list:
                peak_bias_mV = peak_bias * 1e3  # Convert to mV
                if bias_min is not None and bias_max is not None:
                    if bias_min <= peak_bias_mV <= bias_max:
                        peak_biases.append(peak_bias_mV)
                else:
                    peak_biases.append(peak_bias_mV)

        if not peak_biases:
            logging.warning("No peaks detected to plot histogram.")
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.text(0.5, 0.5, 'No Peaks Detected', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xlabel('$V_{b}$ (mV)', fontsize=14)
            ax.set_ylabel('Number of Peaks', fontsize=14)
            plt.tight_layout()
            return fig

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        counts, bins, patches = ax.hist(peak_biases, bins=50, color='tab:orange', edgecolor='black')

        ax.set_xlabel('$V_{b}$ (mV)', fontsize=14)
        ax.set_ylabel('Number of Peaks', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Reduce number of ticks and format tick labels
        ax.locator_params(axis='x', nbins=7)
        ax.locator_params(axis='y', nbins=5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.0f}'))

        # Remove grid and title
        ax.grid(False)

        plt.tight_layout()

        # Store the current histogram data as attributes for saving
        self.current_histogram_counts = counts.copy()
        self.current_histogram_bins = bins.copy()

        if save_fig:
            filename = 'peak_histogram.pdf'
            csv_filename = 'peak_histogram_data.csv'
            try:
                plt.savefig(filename, transparent=True)
                logging.info(f"Peak histogram plot saved as {filename}")
            except Exception as e:
                logging.error(f"Error saving peak histogram plot: {e}")

            # Save histogram data to CSV
            try:
                with open(csv_filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Bias_mV', 'Number_of_Peaks'])
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])
                    for center, count in zip(bin_centers, counts):
                        csvwriter.writerow([center, int(count)])
                logging.info(f"Peak histogram data saved as {csv_filename}")
            except Exception as e:
                logging.error(f"Error saving peak histogram data: {e}")

        return fig

    def save_peak_histogram_csv(self, bias_min, bias_max, filename=None):
        """
        Save the histogram data of number of peaks vs bias voltage within a specified bias window to a CSV file.
        If filename is not provided, use a default name.
        """
        if not hasattr(self, 'current_histogram_bins') or not hasattr(self, 'current_histogram_counts'):
            logging.error("No histogram data available to save. Please generate a histogram plot first.")
            raise AttributeError("Histogram data not found. Run plot_peak_histogram first.")

        if filename is None:
            filename = 'peak_histogram_data.csv'

        try:
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Bias_mV', 'Number_of_Peaks'])
                bin_centers = 0.5 * (self.current_histogram_bins[:-1] + self.current_histogram_bins[1:])
                for center, count in zip(bin_centers, self.current_histogram_counts):
                    csvwriter.writerow([center, int(count)])
            logging.info(f"Peak histogram data saved as {filename}")
        except Exception as e:
            logging.error(f"Error saving peak histogram data: {e}")

    def save_grid_output_csv(self, filename=None):
        """
        Save the current peak counts grid to a CSV file with float64 precision.
        If filename is not provided, use a default name.
        """
        if not hasattr(self, 'current_peak_counts'):
            logging.error("No current_peak_counts available to save. Please generate a plot first.")
            raise AttributeError("current_peak_counts not found. Run plotting_specgrid first.")

        if filename is None:
            filename = 'grid_output_data.csv'

        try:
            # Save with float64 precision using exponential format
            np.savetxt(filename, self.current_peak_counts, delimiter=",", fmt='%.18e')
            logging.info(f"Grid output data saved as {filename}")
        except Exception as e:
            logging.error(f"Error saving grid output data: {e}")

    def save_average_dIdV_csv(self, bias_min, bias_max, filename=None):
        """
        Save the average dI/dV data for the defined bias window to a CSV file with float64 precision.
        If filename is not provided, use a default name.
        """
        if not hasattr(self, 'current_average_spectrum'):
            logging.error("No average_spectrum available to save. Please generate an average spectrum plot first.")
            raise AttributeError("current_average_spectrum not found. Run plotting_average_spectrum first.")

        if filename is None:
            filename = 'average_dIdV_data.csv'

        try:
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Bias_V', 'Average_dI_dV'])
                for bias, avg in zip(self.bias, self.current_average_spectrum):
                    csvwriter.writerow([bias, avg])
            logging.info(f"Average dI/dV data saved as {filename}")
        except Exception as e:
            logging.error(f"Error saving average dI/dV data: {e}")

    # ============================
    # Methods for Subtracting Linear Fit
    # ============================

    def subtract_linear_fit(self):
        """
        Subtract a linear fit from the filtered data and shift to make all dI/dV non-negative.
        """
        logging.info("Applying linear fit subtraction to the filtered data.")

        # Initialize adjusted data array
        lockin_v_adjusted = np.zeros_like(self.lockin_v_raw)

        # Fit and subtract linear trend for each pixel
        for y in range(self.Y_pixel_size):
            for x in range(self.X_pixel_size):
                spectrum = self.lockin_v_raw[:, y, x]
                # Fit a linear trend
                coeffs = np.polyfit(self.bias, spectrum, deg=1)
                linear_trend = np.polyval(coeffs, self.bias)
                # Subtract the linear trend
                lockin_v_adjusted[:, y, x] = spectrum - linear_trend

        # Find the most negative dI/dV across all pixels
        min_dIdV = lockin_v_adjusted.min()
        if min_dIdV < 0:
            shift = abs(min_dIdV) + 1e-12  # Adding a small epsilon
            logging.info(f"Shifting all dI/dV data by {shift} to make them non-negative.")
            lockin_v_adjusted += shift
        else:
            logging.info("No negative dI/dV values found. No shifting needed.")

        # Update the filtered data with adjusted data
        self.lockin_v_filtered = lockin_v_adjusted.copy()

        # Update linear_fit_subtracted flag
        self.linear_fit_subtracted = True

    def reset_linear_fit(self):
        """
        Reset the filtered data to the original raw data before linear fit subtraction.
        """
        logging.info("Resetting filtered data to original raw data.")
        self.lockin_v_filtered = self.lockin_v_raw.copy()

        # Update linear_fit_subtracted flag
        self.linear_fit_subtracted = False

    def save_filtered_data_csv(self, filename=None):
        """
        Save the filtered data (after any processing like linear fit subtraction) to a CSV file with float64 precision.
        If filename is not provided, use a default name.
        """
        if filename is None:
            filename = 'filtered_data.csv'

        try:
            # Save with float64 precision
            np.savetxt(filename, self.lockin_v_filtered, delimiter=",", fmt='%.18e')
            logging.info(f"Filtered data saved as {filename}")
        except Exception as e:
            logging.error(f"Error saving filtered data: {e}")

    def save_raw_data_csv(self, filename=None):
        """
        Save the raw (unfiltered) data to a CSV file with float64 precision.
        If filename is not provided, use a default name.
        """
        if filename is None:
            filename = 'raw_data.csv'

        try:
            # Save with float64 precision
            np.savetxt(filename, self.lockin_v_raw, delimiter=",", fmt='%.18e')
            logging.info(f"Raw data saved as {filename}")
        except Exception as e:
            logging.error(f"Error saving raw data: {e}")

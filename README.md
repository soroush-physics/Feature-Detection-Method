# Feature dectection method

**Author:** Dr. Soroush Arabi  

## Overview
**Feature dectection method** is a Python-based analysis tool for processing and visualizing spectroscopy grid data in the Nanonis `.3ds` format, typically acquired in scanning probe microscopy experiments.  
It provides advanced data filtering, automated peak detection, high-quality visualizations of peak distributions, and convenient export of processed data for further quantitative analysis.

---

## Key Features
- Load spectroscopy grid data from Nanonis `.3ds` files  
- Apply **Savitzky–Golay filtering** for noise reduction  
- Automated peak detection for each pixel spectrum  
- Generate 2D grid maps of peak distributions in a specified bias range  
- Compute average spectra for peak-containing pixels  
- Plot peak occurrence histograms across the grid  
- Export results to **CSV** for reproducibility and downstream analysis  
- Optional Gaussian smoothing of peak maps for enhanced visualization  
- Linear baseline subtraction to correct drift or offset

---

## Requirements
- Python ≥ 3.8  
- [numpy](https://numpy.org/)  
- [matplotlib](https://matplotlib.org/)  
- [scipy](https://scipy.org/)  
- [nanonispy](https://pypi.org/project/nanonispy/) *(for reading `.3ds` files)*

Install dependencies via:
```bash
pip install numpy matplotlib scipy nanonispy
```

---

## Usage Example

### 1. Import and Initialize
```python
from specgrid_processor import SpecgridAnalyser

# Path to your .3ds file
specgrid_path = r"path\to\your\file.3ds"

# Initialize the analyser
analyser = SpecgridAnalyser(specgrid_path)
```

### 2. Apply Filtering and Peak Detection
```python
analyser.apply_sg_filter_and_detect_peaks(
    window_length=5,
    polyorder=2,
    min_distance=2,
    threshold=0.05
)
```

### 3. Plot Peak Distribution Map
```python
fig = analyser.plotting_specgrid(
    bias_min=-0.01,
    bias_max=0.01,
    apply_gaussian_filter=True
)
fig.show()
```

### 4. Plot Average Spectrum for Peak-Containing Pixels
```python
fig = analyser.plotting_average_spectrum(bias_min=-0.01, bias_max=0.01)
fig.show()
```

### 5. Plot Peak Histogram
```python
fig = analyser.plot_peak_histogram(bias_min=-10, bias_max=10)  # mV
fig.show()
```

### 6. Export Data
```python
analyser.save_grid_output_csv("grid_output.csv")
analyser.save_average_dIdV_csv(
    bias_min=-0.01,
    bias_max=0.01,
    filename="average_dIdV.csv"
)
```

---

## Output Files
- **`grid_output_data.csv`** — Peak counts per pixel  
- **`average_dIdV_data.csv`** — Average spectrum within the specified bias range  
- **`peak_histogram_data.csv`** — Histogram of peak occurrences  
- **PDF plots** *(optional)* — When `save_fig=True` is enabled in plotting methods

---

## Complete Workflow Example
```python
# Load the data
analyser = SpecgridAnalyser(r"C:\data\Grid_Spectroscopy001.3ds")

# Process
analyser.apply_sg_filter_and_detect_peaks()

# Visualize
analyser.plotting_specgrid(-0.01, 0.01, save_fig=True)
analyser.plotting_average_spectrum(-0.01, 0.01, save_fig=True)
analyser.plot_peak_histogram(bias_min=-10, bias_max=10, save_fig=True)

# Export results
analyser.save_grid_output_csv()
analyser.save_average_dIdV_csv(-0.01, 0.01)
```

---

## License
All rights reserved. Contact the author for permissions.

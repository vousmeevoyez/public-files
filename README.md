# Fault Detection Pipeline - Local Setup

This project implements a fault detection system for sensor data using neural networks and entropy-based features.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Navigate to `final_databaru_optimized_revisi-2-pisah.ipynb`

3. **Execute cells in order:**
   - Cell 1: Load data
   - Cell 2: Konfigurasi sensor
   - Cell 4: **Fungsi Fault Simulator** (WAJIB dijalankan terlebih dahulu!)
   - Cell 7: **Fungsi Neural Network** (WAJIB dijalankan sebelum run_detection_per_sensor!)
   - Cell 8: Fungsi create_fault_scenario_per_sensor
   - Cell 9: Fungsi run_detection_per_sensor
   - Follow the remaining cells as needed

## Project Structure

- `final_databaru_optimized_revisi-2-pisah.ipynb` - Main notebook with fault detection pipeline
- `tabel_sensor4_generated.csv` - Sensor data file (or will be downloaded from GitHub)
- `requirements.txt` - Python dependencies

## Features

- ✅ Per-sensor fault detection
- ✅ Multiple fault types: drift, spike, bias, hardware
- ✅ Entropy-based feature extraction
- ✅ Neural network classification
- ✅ Time-ordered train/validation/test splits
- ✅ Earliest alarm ranking per sensor
- ✅ Comprehensive visualization

## Notes

- The notebook will automatically download the dataset from GitHub if `tabel_sensor4_generated.csv` is not available locally
- All random seeds are locked for reproducibility
- The pipeline uses time-ordered splits to prevent data leakage




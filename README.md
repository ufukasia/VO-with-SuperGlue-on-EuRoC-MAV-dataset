# Visual Odometry Pipeline with SuperGlue Feature Tracking

![Visual Odometry Plot](vo_vs_gt_plot.png?raw=true)

## Introduction

Welcome to the **Visual Odometry Pipeline** repository! This project implements a streamlined Visual Odometry (VO) system using straightforward Python libraries such as NumPy, OpenCV, Pandas, Matplotlib, and PyTorch, along with the SuperGlue model for advanced feature matching. The pipeline is designed to process visual data, estimate camera poses, and visualize trajectories effectively.

## Features

- **Feature Tracking with SuperGlue:** Utilizes the SuperGlue model for robust feature matching between consecutive frames.
- **Real-time Trajectory Visualization:** Displays estimated and ground truth trajectories in XY, XZ, and YZ planes.
- **Progress Monitoring:** Includes a terminal-based progress bar to track processing status.
- **Interpolated Ground Truth Data:** Automatically generates and uses an interpolated CSV file to align ground truth data with camera timestamps.
- **Comparison Plots:** Produces plots comparing estimated poses with ground truth data.

## Prerequisites

Ensure you have the following installed:

- **Python 3.7 or higher**
- **Git**
- **CUDA (optional):** For GPU acceleration with PyTorch (recommended for SuperGlue)

## Installation

1. **Clone the Repository**

   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/ufukasia/VO-with-SuperGlue-on-EuRoC-MAV-dataset.git)
   cd VO-with-SuperGlue-on-EuRoC-MAV-dataset
   ```


   **Note:** Ensure PyTorch with CUDA support is installed for optimal performance.

## Dataset Preparation

This pipeline is compatible with datasets structured similarly to the EuRoC MAV dataset. Ensure your dataset follows the directory structure below:



### Steps to Prepare the Dataset

1. **Download the Dataset:**

   Obtain the dataset (e.g., EuRoC MAV) and place it in the `dataset/` directory following the structure above.

2. **Verify File Paths:**

   Ensure that the camera and ground truth CSV files are correctly placed in their respective directories.

3. **Check Image Files:**

   Camera images should be named using their corresponding timestamps (e.g., `1234567890.png`) and placed in the `cam0/data/` directory.

4. **Update Dataset Path (If Necessary):**

   If your dataset is located elsewhere, update the `dataset_path` variable in the `superglue_EurocMAV.py` script accordingly:

   ```python
   dataset_path = Path("path/to/your/dataset/MH_01_easy/mav0/")
   ```

## Usage

Run the Visual Odometry pipeline by executing the main script. The script preprocesses ground truth data, performs feature tracking, estimates poses, and visualizes trajectories.

```bash
python superglue_EurocMAV.py
```

### Script Workflow

1. **Preprocess Ground Truth Data:**

   The script generates an interpolated CSV file (`imu_with_interpolated_groundtruth.csv`) to align ground truth data with camera timestamps.

2. **Feature Tracking and Pose Estimation:**

   Utilizes SuperGlue for feature matching and estimates camera poses based on matched features.

3. **Trajectory Visualization:**

   Displays real-time trajectories and saves comparison plots.

## Outputs

- **Interpolated Ground Truth Data:**
  
  - `imu_with_interpolated_groundtruth.csv` in the `imu0/` directory.

- **Real-time Trajectory Visualizations:**
  
  - Three OpenCV windows displaying trajectories in XY, XZ, and YZ planes.
  - A window showing the current camera frame.






## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- **[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork):** For providing an advanced feature matching framework.
- **[OpenCV](https://opencv.org/):** For image processing and computer vision functionalities.
- **[PyTorch](https://pytorch.org/):** For deep learning model support.
- **[tqdm](https://github.com/tqdm/tqdm):** For the progress bar implementation.
- **[EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets):** For providing the dataset used in this pipeline.

---

*Developed by Your Name. Feel free to reach out for any questions or collaborations.*

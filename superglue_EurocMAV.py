import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt  # For plotting
from pathlib import Path  # For handling file paths
import matplotlib
import os
from tqdm import tqdm  # Progress bar
matplotlib.use('TkAgg')  # Backend suitable for real-time display

# SuperGlue imports
from models.matching import Matching
from models.utils import frame2tensor
import torch

dataset_path = Path("MH_01_easy/mav0/")

def preprocess_imu_data(dataset_path, cam_to_imu_timeshift=5.63799926987e-05):
    """
    Preprocess IMU data by applying a time shift to ground truth data and interpolating it to align with IMU timestamps.
    
    Parameters:
    - dataset_path: Path object pointing to the dataset directory.
    - cam_to_imu_timeshift: Time shift in seconds to align camera and IMU data.
    
    Creates:
    - imu_with_interpolated_groundtruth.csv in imu0 directory.
    """
    try:
        imu_file = dataset_path / 'imu0/data.csv'
        groundtruth_file = dataset_path / 'state_groundtruth_estimate0/data.csv'
        
        # Check if files exist
        if not imu_file.exists():
            raise FileNotFoundError(f"IMU data file not found at {imu_file}")
        if not groundtruth_file.exists():
            raise FileNotFoundError(f"Ground truth data file not found at {groundtruth_file}")
        
        # Read data
        imu_df = pd.read_csv(imu_file)
        groundtruth_df = pd.read_csv(groundtruth_file)
        
        # Strip whitespace from column names
        groundtruth_df.columns = groundtruth_df.columns.str.strip()
        imu_df.columns = imu_df.columns.str.strip()
        
        # Print column names for debugging
        print("IMU Data Columns:", imu_df.columns.tolist())
        print("Ground Truth Data Columns:", groundtruth_df.columns.tolist())
        
        # Time shift correction
        timeshift_ns = int(cam_to_imu_timeshift * 1e9)
        print(f"Applying time shift of {cam_to_imu_timeshift} seconds ({timeshift_ns} nanoseconds)")
        
        # Check required columns
        if '#timestamp' not in groundtruth_df.columns:
            raise KeyError("Ground truth data does not contain '#timestamp' column.")
        if '#timestamp [ns]' not in imu_df.columns:
            raise KeyError("IMU data does not contain '#timestamp [ns]' column.")
        
        # Apply time shift to ground truth timestamps
        groundtruth_df['#timestamp'] = groundtruth_df['#timestamp'].apply(lambda x: x + timeshift_ns)
        
        # Set timestamp as index and sort
        groundtruth_df.set_index('#timestamp', inplace=True)
        groundtruth_df.sort_index(inplace=True)
        
        # Select numeric columns excluding timestamp
        numeric_cols = groundtruth_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != '#timestamp']
        
        print(f"Numeric columns to interpolate: {numeric_cols}")
        
        # Sort IMU data by timestamp
        imu_df.sort_values('#timestamp [ns]', inplace=True)
        
        # Interpolate each numeric column
        for col in numeric_cols:
            if col in groundtruth_df.columns:
                print(f"Interpolating column: {col}")
                imu_df[col] = np.interp(
                    imu_df['#timestamp [ns]'],
                    groundtruth_df.index.values,
                    groundtruth_df[col].values
                )
            else:
                print(f"Warning: Column '{col}' not found in ground truth data. Skipping interpolation for this column.")
        
        # Save interpolated data
        output_file = dataset_path / 'imu0/imu_with_interpolated_groundtruth.csv'
        imu_df.to_csv(output_file, index=False)
        print(f"Preprocessed IMU data saved to {output_file}")
        
        # Print time shift applied
        print(f"Applied timeshift correction: {timeshift_ns} ns ({cam_to_imu_timeshift} s)")
        
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise

# Frame processing stages
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

# Body-sensor transformation matrix (for Euroc MAV, T_BS)
T_BS = np.array([
    [ 0.01517066, -0.99983694,  0.00979558, -0.01638528],
    [ 0.99965712,  0.01537559,  0.02119505, -0.06812726],
    [-0.02134221,  0.00947067,  0.99972737,  0.00395795],
    [ 0.0,          0.0,         0.0,         1.0        ]
])

def featureTracking(image_ref, image_cur, px_ref):
    """
    Performs feature matching between a reference image and the current image using SuperGlue.

    Parameters:
    - image_ref: Reference image (previous frame).
    - image_cur: Current image (new frame).
    - px_ref: Reference keypoints (unused but kept for function signature consistency).

    Returns:
    - kp1: Matched keypoints from the reference image.
    - kp2: Matched keypoints from the current image.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert images to tensors
    inp0 = frame2tensor(image_ref, device)
    inp1 = frame2tensor(image_cur, device)

    # Access the global VO instance's matching object
    global vo_global
    output = vo_global.matching({'image0': inp0, 'image1': inp1})

    # Extract keypoints and matches
    kpts0 = output['keypoints0'][0].cpu().numpy()
    kpts1 = output['keypoints1'][0].cpu().numpy()
    matches = output['matches0'][0].cpu().numpy()
    valid = matches > -1
    kp1 = kpts0[valid]  # Matched keypoints in reference image
    kp2 = kpts1[matches[valid]]  # Corresponding matched keypoints in current image

    return kp1.astype(np.float32), kp2.astype(np.float32)

def rotation_matrix_to_euler_angles(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw) following the ZYX order.

    Parameters:
    - R: 3x3 rotation matrix.

    Returns:
    - Array of Euler angles [roll, pitch, yaw] in radians.
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])  # Rotation around X-axis
        pitch = np.arctan2(-R[2,0], sy)    # Rotation around Y-axis
        yaw = np.arctan2(R[1,0], R[0,0])   # Rotation around Z-axis
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw], dtype=np.float64)

def euler_angles_to_rotation_matrix(euler):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix following the ZYX order.

    Parameters:
    - euler: Array of Euler angles [roll, pitch, yaw] in radians.

    Returns:
    - 3x3 rotation matrix.
    """
    roll, pitch, yaw = euler
    Rz = np.array([
        [ np.cos(yaw), -np.sin(yaw), 0],
        [ np.sin(yaw),  np.cos(yaw), 0],
        [          0,           0,   1]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ], dtype=np.float64)

    Rx = np.array([
        [1,           0,            0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ], dtype=np.float64)

    return Rz @ Ry @ Rx

def clamp_euler_angles(euler_old, euler_new, max_deg=5.0):
    """
    Clamps the change in Euler angles to ensure continuity between frames.

    Parameters:
    - euler_old: Previous Euler angles [roll, pitch, yaw] in radians.
    - euler_new: Newly estimated Euler angles [roll, pitch, yaw] in radians.
    - max_deg: Maximum allowed change in degrees for each angle.

    Returns:
    - Clamped Euler angles ensuring smooth transitions.
    """
    max_rad = np.deg2rad(max_deg)
    diff = euler_new - euler_old

    # Normalize the angle difference to be within [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    clamped = euler_old.copy()
    for i in range(3):
        if abs(diff[i]) <= max_rad:
            clamped[i] = euler_new[i]
        else:
            if diff[i] > 0:
                clamped[i] = euler_old[i] + max_rad
            else:
                clamped[i] = euler_old[i] - max_rad
    return clamped

class VisualOdometry:
    def __init__(self, cam, gt_data, cam_data):
        """
        Initializes the Visual Odometry system.

        Parameters:
        - cam: PinholeCamera object containing camera parameters.
        - gt_data: Ground truth data as a pandas DataFrame.
        - cam_data: Camera data as a pandas DataFrame.
        """
        self.frame_stage = STAGE_FIRST_FRAME
        self.cam = cam
        self.new_frame = None
        self.last_frame = None

        # Initialize current translation with the first ground truth position
        first_gt = gt_data.iloc[0]
        self.cur_t = np.array([
            first_gt['p_RS_R_x [m]'],
            first_gt['p_RS_R_y [m]'],
            first_gt['p_RS_R_z [m]']
        ]).reshape(3, 1)

        # Convert initial quaternion to rotation matrix
        q_w = first_gt['q_RS_w []']
        q_x = first_gt['q_RS_x []']
        q_y = first_gt['q_RS_y []']
        q_z = first_gt['q_RS_z []']
        R_imu = self.quaternion_to_rotation_matrix(q_w, q_x, q_y, q_z)

        self.cur_R = R_imu.copy()
        self.prev_euler = rotation_matrix_to_euler_angles(self.cur_R)

        # Initialize current velocity with the first ground truth velocity
        self.cur_vel = np.array([
            first_gt['v_RS_R_x [m s^-1]'],
            first_gt['v_RS_R_y [m s^-1]'],
            first_gt['v_RS_R_z [m s^-1]']
        ]).reshape(3, 1)

        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        self.trueX, self.trueY, self.trueZ = 0, 0, 0

        # Initialize FAST feature detector for the first frame
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.gt_data = gt_data
        self.cam_data = cam_data

        # SuperGlue configuration
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.matching = Matching(config).eval().to(device)

    def quaternion_to_rotation_matrix(self, w, x, y, z):
        """
        Converts a quaternion to a rotation matrix.

        Parameters:
        - w, x, y, z: Components of the quaternion.

        Returns:
        - 3x3 rotation matrix.
        """
        R = np.zeros((3, 3))
        R[0, 0] = 1 - 2 * (y**2 + z**2)
        R[0, 1] = 2 * (x * y - z * w)
        R[0, 2] = 2 * (x * z + y * w)

        R[1, 0] = 2 * (x * y + z * w)
        R[1, 1] = 1 - 2 * (x**2 + z**2)
        R[1, 2] = 2 * (y * z - x * w)

        R[2, 0] = 2 * (x * z - y * w)
        R[2, 1] = 2 * (y * z + x * w)
        R[2, 2] = 1 - 2 * (x**2 + y**2)
        return R

    def getAbsoluteScale(self, frame_id):
        """
        Retrieves the absolute scale between consecutive frames using ground truth data.

        Parameters:
        - frame_id: Current frame index.

        Returns:
        - Scale factor as a float.
        """
        if frame_id < 1:
            return 0
        curr_timestamp = self.cam_data.iloc[frame_id]['#timestamp [ns]']
        prev_timestamp = self.cam_data.iloc[frame_id - 1]['#timestamp [ns]']

        curr_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == curr_timestamp]
        prev_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == prev_timestamp]

        if len(curr_gt) == 0 or len(prev_gt) == 0:
            return 0

        curr_gt = curr_gt.iloc[0]
        prev_gt = prev_gt.iloc[0]

        x_prev = prev_gt['p_RS_R_x [m]']
        y_prev = prev_gt['p_RS_R_y [m]']
        z_prev = prev_gt['p_RS_R_z [m]']

        x = curr_gt['p_RS_R_x [m]']
        y = curr_gt['p_RS_R_y [m]']
        z = curr_gt['p_RS_R_z [m]']

        self.trueX, self.trueY, self.trueZ = x, y, z
        scale = np.sqrt((x - x_prev)**2 + (y - y_prev)**2 + (z - z_prev)**2)
        return scale

    def processFirstFrame(self):
        """
        Processes the first frame by detecting features and setting the reference keypoints.
        """
        # Detect features in the first frame using FAST
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        """
        Processes the second frame by matching features and estimating the initial pose.
        """
        # Match features between the first and second frames using SuperGlue
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, 
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=0.1)
        if E is None:
            print("Essential matrix not found in second frame.")
            self.frame_stage = STAGE_FIRST_FRAME
            return
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp)
        
        # Assume scale = 1 for the initial estimation
        T_cam = np.eye(4)
        T_cam[:3, :3] = R
        T_cam[:3, 3] = t.reshape(3)

        # Correct the pose using the body-sensor transformation
        T_BS_inv = np.linalg.inv(T_BS)
        T_cam_corrected = T_BS @ T_cam @ T_BS_inv
        R_cam = T_cam_corrected[:3, :3]

        # Update the current rotation
        new_R = self.cur_R @ R_cam

        # Clamp the Euler angles to ensure smooth transitions
        new_euler = rotation_matrix_to_euler_angles(new_R)
        clamped_euler = clamp_euler_angles(self.prev_euler, new_euler, max_deg=5.0)
        final_R = euler_angles_to_rotation_matrix(clamped_euler)

        self.cur_R = final_R
        self.prev_euler = clamped_euler

        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        """
        Processes subsequent frames by matching features, estimating pose, and updating the current state.

        Parameters:
        - frame_id: Current frame index.
        """
        # Match features between the last and current frames using SuperGlue
        self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref,
                                       focal=self.focal, pp=self.pp,
                                       method=cv2.RANSAC, prob=0.999, threshold=0.7)
        if E is None:
            print(f"Essential matrix not found at frame {frame_id}.")
            return
        _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref,
                                        focal=self.focal, pp=self.pp)

        # Retrieve the absolute scale from ground truth
        absolute_scale = self.getAbsoluteScale(frame_id)

        if absolute_scale > 0.001:
            # Construct the pose transformation matrix
            T_cam = np.eye(4)
            T_cam[:3, :3] = R
            T_cam[:3, 3] = (absolute_scale * t).reshape(3)

            # Correct the pose using the body-sensor transformation
            T_BS_inv = np.linalg.inv(T_BS)
            T_cam_corrected = T_BS @ T_cam @ T_BS_inv

            R_cam = T_cam_corrected[:3, :3]
            t_cam = T_cam_corrected[:3, 3].reshape(3, 1)

            # Update the current translation and rotation
            new_t = self.cur_t + np.dot(self.cur_R, t_cam)
            new_R = self.cur_R @ R_cam

            # Clamp the Euler angles to ensure smooth transitions
            new_euler = rotation_matrix_to_euler_angles(new_R)
            clamped_euler = clamp_euler_angles(self.prev_euler, new_euler, max_deg=5.0)
            final_R = euler_angles_to_rotation_matrix(clamped_euler)

            self.cur_R = final_R
            self.cur_t = new_t
            self.prev_euler = clamped_euler

            # Update current velocity from ground truth
            curr_timestamp = self.cam_data.iloc[frame_id]['#timestamp [ns]']
            curr_gt = self.gt_data[self.gt_data['#timestamp [ns]'] == curr_timestamp]
            if len(curr_gt) > 0:
                curr_gt = curr_gt.iloc[0]
                self.cur_vel = np.array([
                    curr_gt['v_RS_R_x [m s^-1]'],
                    curr_gt['v_RS_R_y [m s^-1]'],
                    curr_gt['v_RS_R_z [m s^-1]']
                ]).reshape(3, 1)

        # Update the reference keypoints for the next frame
        self.px_ref = self.px_cur

    def update(self, img, frame_id):
        """
        Updates the VO system with a new frame.

        Parameters:
        - img: Current grayscale image frame.
        - frame_id: Current frame index.
        """
        # Ensure the input image is grayscale and matches camera dimensions
        if not (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width):
            raise ValueError("Frame dimensions do not match or image is not grayscale.")
        self.new_frame = img

        # Process the frame based on the current stage
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()

        self.last_frame = self.new_frame

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        """
        Represents a pinhole camera model with optional distortion parameters.

        Parameters:
        - width: Image width in pixels.
        - height: Image height in pixels.
        - fx, fy: Focal lengths.
        - cx, cy: Principal point coordinates.
        - k1, k2, p1, p2, k3: Distortion coefficients.
        """
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]
        self.distortion = any(abs(p) > 1e-7 for p in self.d)

    def undistort_image(self, img):
        """
        Undistorts an image using the camera's distortion parameters.

        Parameters:
        - img: Input distorted image.

        Returns:
        - Undistorted image.
        """
        if not self.distortion:
            return img
        camera_matrix = np.array([[self.fx, 0, self.cx],
                                  [0, self.fy, self.cy],
                                  [0,       0,       1]], dtype=np.float32)
        dist_coeffs = np.array(self.d, dtype=np.float32)
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        return undistorted

def main():
    """
    Main function to execute the Visual Odometry pipeline.
    """
    # Preprocess IMU data
    print("Starting IMU data preprocessing...")
    preprocess_imu_data(dataset_path)
    print("IMU data preprocessing completed.\n")

    # Define camera parameters for Euroc MAV
    cam = PinholeCamera(
        width=752, height=480,
        fx=458.654, fy=457.296,
        cx=367.215, cy=248.375,
        k1=-0.28340811, k2=0.07395907, p1=0.00019359, p2=1.76187114e-05
    )

    # Load ground truth and camera data
    gt_file = dataset_path / "imu0/imu_with_interpolated_groundtruth.csv"
    cam_file = dataset_path / "cam0/data.csv"

    if not gt_file.exists():
        raise FileNotFoundError(f"Interpolated ground truth data file not found at {gt_file}. Please ensure preprocessing is successful.")

    if not cam_file.exists():
        raise FileNotFoundError(f"Camera data file not found at {cam_file}.")

    gt_data = pd.read_csv(gt_file)
    cam_data = pd.read_csv(cam_file)

    vo = VisualOdometry(cam, gt_data, cam_data)

    # Set the global VO reference for featureTracking
    global vo_global
    vo_global = vo

    # Initialize empty images for trajectory visualization
    traj_xy = np.zeros((800, 800, 3), dtype=np.uint8)
    traj_xz = np.zeros((800, 800, 3), dtype=np.uint8)
    traj_yz = np.zeros((800, 800, 3), dtype=np.uint8)

    # Lists to store Euler angles and velocities for plotting
    predicted_euler_list = []  # VO estimated Euler angles (roll, pitch, yaw)
    gt_euler_list = []         # Ground truth Euler angles
    predicted_vel_list = []    # VO estimated velocities
    gt_vel_list = []           # Ground truth velocities

    # Variables to center the trajectory plots
    initial_position_set = False
    initial_x, initial_y, initial_z = 0, 0, 0
    center = 400  # Center of the trajectory image
    scale = 50  # Scale factor for plotting

    # Initialize tqdm progress bar
    total_frames = len(cam_data)
    print("Starting frame processing with progress bar...")
    for idx, row in tqdm(cam_data.iterrows(), total=total_frames, desc="Processing Frames"):
        timestamp = row['#timestamp [ns]']
        img_path = dataset_path / f"cam0/data/{timestamp}.png"
        img = cv2.imread(str(img_path), 0)  # Read as grayscale
        if img is None:
            print(f"Error: Could not read image at path {img_path}. Skipping this frame.")
            continue

        undistorted_img = cam.undistort_image(img)
        vo.update(undistorted_img, idx)

        # Skip the first two frames as they are used for initialization
        if idx <= 2:
            continue

        cur_t = vo.cur_t
        x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]

        # Set the initial position to center the plots
        if not initial_position_set:
            initial_x, initial_y, initial_z = x, y, z
            initial_position_set = True

        # Append VO estimated Euler angles and velocities
        vo_euler = rotation_matrix_to_euler_angles(vo.cur_R)  # Clamped rotation matrix
        vo_vel = vo.cur_vel.reshape(-1)  # [vx, vy, vz]
        predicted_euler_list.append(vo_euler)
        predicted_vel_list.append(vo_vel)

        # Retrieve ground truth Euler angles and velocities
        curr_gt = gt_data[gt_data['#timestamp [ns]'] == timestamp]
        if len(curr_gt) > 0:
            curr_gt = curr_gt.iloc[0]
            # Convert ground truth quaternion to rotation matrix
            qw, qx = curr_gt['q_RS_w []'], curr_gt['q_RS_x []']
            qy, qz = curr_gt['q_RS_y []'], curr_gt['q_RS_z []']
            R_gt = vo.quaternion_to_rotation_matrix(qw, qx, qy, qz)
            gt_euler = rotation_matrix_to_euler_angles(R_gt)

            gt_vel = np.array([
                curr_gt['v_RS_R_x [m s^-1]'],
                curr_gt['v_RS_R_y [m s^-1]'],
                curr_gt['v_RS_R_z [m s^-1]']
            ], dtype=np.float32)
        else:
            gt_euler = np.array([0, 0, 0], dtype=np.float32)
            gt_vel = np.array([0, 0, 0], dtype=np.float32)

        gt_euler_list.append(gt_euler)
        gt_vel_list.append(gt_vel)

        # Calculate positions for trajectory plots (scaled and centered)
        draw_xy_x = int((x - initial_x) * scale) + center
        draw_xy_y = int((y - initial_y) * scale) + center

        draw_xz_x = int((x - initial_x) * scale) + center
        draw_xz_y = int((z - initial_z) * scale) + center

        draw_yz_x = int((y - initial_y) * scale) + center
        draw_yz_y = int((z - initial_z) * scale) + center

        # Calculate ground truth positions for plotting
        true_xy_x = int((vo.trueX - initial_x) * scale) + center
        true_xy_y = int((vo.trueY - initial_y) * scale) + center

        true_xz_x = int((vo.trueX - initial_x) * scale) + center
        true_xz_y = int((vo.trueZ - initial_z) * scale) + center

        true_yz_x = int((vo.trueY - initial_y) * scale) + center
        true_yz_y = int((vo.trueZ - initial_z) * scale) + center

        # Draw estimated trajectory in XY plane
        if 0 <= draw_xy_x < traj_xy.shape[1] and 0 <= draw_xy_y < traj_xy.shape[0]:
            cv2.circle(traj_xy, (draw_xy_x, draw_xy_y), 1,
                       (int(idx * 255 / len(cam_data)), 255 - int(idx * 255 / len(cam_data)), 0), 1)
        # Draw ground truth trajectory in XY plane
        if 0 <= true_xy_x < traj_xy.shape[1] and 0 <= true_xy_y < traj_xy.shape[0]:
            cv2.circle(traj_xy, (true_xy_x, true_xy_y), 1, (0, 0, 255), 2)

        # Draw estimated trajectory in XZ plane
        if 0 <= draw_xz_x < traj_xz.shape[1] and 0 <= draw_xz_y < traj_xz.shape[0]:
            cv2.circle(traj_xz, (draw_xz_x, draw_xz_y), 1,
                       (int(idx * 255 / len(cam_data)), 255 - int(idx * 255 / len(cam_data)), 0), 1)
        # Draw ground truth trajectory in XZ plane
        if 0 <= true_xz_x < traj_xz.shape[1] and 0 <= true_xz_y < traj_xz.shape[0]:
            cv2.circle(traj_xz, (true_xz_x, true_xz_y), 1, (0, 0, 255), 2)

        # Draw estimated trajectory in YZ plane
        if 0 <= draw_yz_x < traj_yz.shape[1] and 0 <= draw_yz_y < traj_yz.shape[0]:
            cv2.circle(traj_yz, (draw_yz_x, draw_yz_y), 1,
                       (int(idx * 255 / len(cam_data)), 255 - int(idx * 255 / len(cam_data)), 0), 1)
        # Draw ground truth trajectory in YZ plane
        if 0 <= true_yz_x < traj_yz.shape[1] and 0 <= true_yz_y < traj_yz.shape[0]:
            cv2.circle(traj_yz, (true_yz_x, true_yz_y), 1, (0, 0, 255), 2)

        # Display coordinate information on the XY trajectory image
        cv2.rectangle(traj_xy, (10, 20), (750, 60), (0, 0, 0), -1)
        text_xy = f"XY Coord: x={x:.2f} y={y:.2f}"
        cv2.putText(traj_xy, text_xy, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # Display coordinate information on the XZ trajectory image
        cv2.rectangle(traj_xz, (10, 20), (750, 60), (0, 0, 0), -1)
        text_xz = f"XZ Coord: x={x:.2f} z={z:.2f}"
        cv2.putText(traj_xz, text_xz, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # Display coordinate information on the YZ trajectory image
        cv2.rectangle(traj_yz, (10, 20), (750, 60), (0, 0, 0), -1)
        text_yz = f"YZ Coord: y={y:.2f} z={z:.2f}"
        cv2.putText(traj_yz, text_yz, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # Update the real-time trajectory and camera display windows
        cv2.imshow('Trajectory XY', traj_xy)
        cv2.imshow('Trajectory XZ', traj_xz)
        cv2.imshow('Trajectory YZ', traj_yz)
        cv2.imshow('Camera', undistorted_img)
        key = cv2.waitKey(1)
        if key == 27:  # Press ESC to exit early
            print("Early termination by user.")
            break

    # Plotting after processing all frames
    # Convert lists to numpy arrays for easier manipulation
    pred_euler = np.array(predicted_euler_list)  # Shape (N, 3), in radians
    gt_euler = np.array(gt_euler_list)           # Shape (N, 3), in radians
    pred_vel = np.array(predicted_vel_list)      # Shape (N, 3)
    gt_vel = np.array(gt_vel_list)               # Shape (N, 3)

    # Unwrap angles to prevent discontinuities in plotting
    pred_euler_unwrapped = pred_euler.copy()
    gt_euler_unwrapped = gt_euler.copy()

    for i in range(3):
        pred_euler_unwrapped[:, i] = np.unwrap(pred_euler_unwrapped[:, i])
        gt_euler_unwrapped[:, i] = np.unwrap(gt_euler_unwrapped[:, i])

    # Convert radians to degrees for better interpretability
    pred_euler_deg = np.rad2deg(pred_euler_unwrapped)
    gt_euler_deg = np.rad2deg(gt_euler_unwrapped)

    # Create subplots for Euler angles and velocities comparison
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("VO vs Ground Truth: Euler Angles and Velocity Comparison", fontsize=16)

    # Titles for Euler angles
    titles_euler = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    for i in range(3):
        axs[0, i].plot(pred_euler_deg[:, i], label='Estimated', color='blue')
        axs[0, i].plot(gt_euler_deg[:, i], label='Ground Truth', color='orange')
        axs[0, i].set_title(titles_euler[i])
        axs[0, i].legend()
        axs[0, i].grid(True)

    # Titles for velocities
    titles_vel = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
    for i in range(3):
        axs[1, i].plot(pred_vel[:, i], label='Estimated', color='green')
        axs[1, i].plot(gt_vel[:, i], label='Ground Truth', color='red')
        axs[1, i].set_title(titles_vel[i])
        axs[1, i].legend()
        axs[1, i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('vo_vs_gt_plot.png')  # Save the final comparison plot
    plt.close()  # Close the plot to free memory
    print("Plot saved as 'vo_vs_gt_plot.png'.")

if __name__ == "__main__":
    main()

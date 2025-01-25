import os
import cv2
from datetime import datetime
import shutil  # 폴더 덮어쓰기용

def parse_timestamps(file_path):
    """Parse the timestamps.txt file into a list of (timestamp, filename)."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [(datetime.strptime(line.strip(), "%Y%m%d_%H%M%S_%f"), line.strip()) for line in lines]

def find_closest_pairs(realsense_timestamps, usb_timestamps):
    """Find the closest matching pairs of realsense and usb timestamps."""
    matched_pairs = []
    i, j = 0, 0
    while i < len(realsense_timestamps) and j < len(usb_timestamps):
        rs_time, rs_filename = realsense_timestamps[i]
        usb_time, usb_filename = usb_timestamps[j]
        time_diff = abs((rs_time - usb_time).total_seconds())
        
        if i + 1 < len(realsense_timestamps) and abs((realsense_timestamps[i + 1][0] - usb_time).total_seconds()) < time_diff:
            i += 1
        elif j + 1 < len(usb_timestamps) and abs((usb_timestamps[j + 1][0] - rs_time).total_seconds()) < time_diff:
            j += 1
        else:
            matched_pairs.append((rs_filename, usb_filename))
            i += 1
            j += 1
    return matched_pairs

def sync_usb_to_realsense(realsense_path, usb_path):
    """Synchronize USB images to match RealSense images and overwrite USB folder."""
    # Parse timestamps
    realsense_timestamps = parse_timestamps(os.path.join(realsense_path, "timestamps.txt"))
    usb_timestamps = parse_timestamps(os.path.join(usb_path, "timestamps.txt"))

    # Find matched pairs
    matched_pairs = find_closest_pairs(realsense_timestamps, usb_timestamps)

    # Create temporary directory for synchronized USB images
    temp_usb_path = os.path.join(usb_path, "temp")
    os.makedirs(temp_usb_path, exist_ok=True)

    # Copy matched USB images to the temporary directory
    new_timestamps = []
    for _, usb_filename in matched_pairs:
        usb_image_path = os.path.join(usb_path, f"{usb_filename}.jpg")
        if os.path.exists(usb_image_path):
            shutil.copy(usb_image_path, temp_usb_path)
            new_timestamps.append(usb_filename)

    # Overwrite the USB folder with the synchronized images
    for file in os.listdir(usb_path):
        if file.endswith(".jpg") or file == "timestamps.txt":
            os.remove(os.path.join(usb_path, file))
    for file in os.listdir(temp_usb_path):
        shutil.move(os.path.join(temp_usb_path, file), usb_path)
    os.rmdir(temp_usb_path)

    # Update timestamps.txt
    with open(os.path.join(usb_path, "timestamps.txt"), "w") as f:
        for timestamp in new_timestamps:
            f.write(f"{timestamp}\n")

    print(f"[INFO] USB folder synchronized with RealSense at: {usb_path}")

# Example usage
sync_usb_to_realsense(
    realsense_path="./record_save_repo/20250125_180012/realsense",
    usb_path="./record_save_repo/20250125_180012/usb_cam"
)

# from datetime import datetime

# # Example: Load timestamps from both folders
# realsense_timestamps = [datetime.strptime(line.strip(), "%Y%m%d_%H%M%S_%f") for line in open("./record_save_repo/20250125_180012//realsense/timestamps.txt")]
# usb_timestamps = [datetime.strptime(line.strip(), "%Y%m%d_%H%M%S_%f") for line in open("./record_save_repo/20250125_180012//usb_cam/timestamps.txt")]

# # Compute time differences
# time_differences = [abs((rs - usb).total_seconds()) for rs, usb in zip(realsense_timestamps, usb_timestamps)]

# # Print maximum, minimum, and average time difference
# print(f"Max diff: {max(time_differences)} seconds")
# print(f"Min diff: {min(time_differences)} seconds")
# print(f"Average diff: {sum(time_differences)/len(time_differences)} seconds")
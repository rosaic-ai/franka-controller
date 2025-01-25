import os
import cv2


def create_individual_videos(image_path, output_video_path, fps=30):
    """
    Create a video from images in the given path.

    Args:
        image_path (str): Path to the folder containing images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
    image_files = sorted([f for f in os.listdir(image_path) if f.endswith('.jpg')])

    if not image_files:
        print(f"[ERROR] No images found in {image_path}.")
        return

    # Read the first image to determine frame size
    first_image = cv2.imread(os.path.join(image_path, image_files[0]))
    if first_image is None:
        print(f"[ERROR] Could not read images in {image_path}.")
        return

    height, width = first_image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"[INFO] Creating video: {output_video_path}")
    for image_file in image_files:
        image = cv2.imread(os.path.join(image_path, image_file))
        if image is None:
            print(f"[WARNING] Missing image: {image_file}. Skipping...")
            continue
        video_writer.write(image)

    video_writer.release()
    print(f"[INFO] Video saved to {output_video_path}")


# Example usage
# Adjust FPS to match the original data capture rate
original_fps = 10  # Replace with your actual data capture FPS if different

create_individual_videos(
    image_path="./record_save_repo/20250125_180012/realsense",
    output_video_path="./record_save_repo/20250125_180012/realsense_video.mp4",
    fps=original_fps
)

create_individual_videos(
    image_path="./record_save_repo/20250125_180012/usb_cam",
    output_video_path="./record_save_repo/20250125_180012/usb_cam_video.mp4",
    fps=original_fps
)

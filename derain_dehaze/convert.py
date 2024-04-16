import cv2
import os

# Function to convert a video to a sequence of images and save them in a folder with the video's name
def video_to_images(video_path, output_folder):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_folder, video_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))  # Get the frame rate of the original video

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()

    print(f"{frame_count} frames extracted to {output_folder} (Frame size: {frame_width}x{frame_height}, Frame rate: {frame_rate})")

# Function to convert a sequence of images back to a video with the same frame rate and size
def images_to_video(image_folder, output_video_path, frame_rate, frame_size):
    image_files = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

    for image_file in image_files:
        print(image_file)
        frame = cv2.imread(image_file)
        out.write(frame)

    out.release()

    print(f"Video created at {output_video_path} (Frame size: {frame_size[0]}x{frame_size[1]}, Frame rate: {frame_rate})")

# Example usage
video_path = '/home/sumanth/Desktop/drdo_demo_videos/v2/original/fog_with_low_light.avi'
output_folder = '/home/sumanth/Desktop/drdo_demo_videos/v2/derain_pred_images/fog_low_lite'

# Convert video to images and create a folder for the video
# video_to_images(video_path, output_folder)

# Get the frame size and frame rate from the original video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))
cap.release()
print(frame_rate)

# Convert images back to video with the same frame rate and size
images_to_video(output_folder, "/home/sumanth/Desktop/drdo_demo_videos/v2/only_derain/fog_with_low_light.avi", 4, (frame_width, frame_height))


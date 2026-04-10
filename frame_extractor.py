import cv2
import numpy as np
import sys

# Constants
VIDEO_PATH = "Top view footage of the parking lot - Top Shared Videos (1080p, h264).mp4"

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        sys.exit(1)

    # Extract Background Reference (Frame 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, f1 = cap.read()

    # Extract Current State (Frame 30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret2, f2 = cap.read()

    if ret1 and ret2:
        f1.tofile("frame1.bin")
        f2.tofile("frame2.bin")
        print(f"Successfully saved frame1.bin and frame2.bin {f1.shape}")
    else:
        print("Error: Failed to retrieve frames from video stream.")

    cap.release()

if __name__ == "__main__":
    main()

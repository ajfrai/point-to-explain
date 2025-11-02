#!/usr/bin/env python3
"""
Camera Reader Module for Point-to-Explain
Supports Arduino-style cameras on NVIDIA Jetson Nano including:
- CSI cameras (Raspberry Pi Camera Module v2, IMX219, etc.)
- USB cameras
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CameraReader:
    """
    Camera reader class for capturing frames from various camera types.
    Optimized for NVIDIA Jetson Nano with hardware acceleration.
    """

    def __init__(
        self,
        camera_type: str = "csi",
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        framerate: int = 30,
        flip_method: int = 0
    ):
        """
        Initialize the camera reader.

        Args:
            camera_type: Type of camera - "csi" for CSI camera, "usb" for USB camera
            camera_id: Camera device ID (0 for first camera, 1 for second, etc.)
            width: Frame width in pixels
            height: Frame height in pixels
            framerate: Frames per second
            flip_method: Flip method for CSI cameras (0-7)
                        0: No flip
                        2: Rotate 180 degrees
                        3: Rotate 90 degrees clockwise
                        4: Rotate 90 degrees counter-clockwise
        """
        self.camera_type = camera_type.lower()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.framerate = framerate
        self.flip_method = flip_method
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def _get_csi_pipeline(self) -> str:
        """
        Generate GStreamer pipeline string for CSI camera.
        Uses hardware-accelerated nvarguscamerasrc for Jetson.

        Returns:
            GStreamer pipeline string
        """
        return (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self.framerate}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! "
            f"appsink"
        )

    def open(self) -> bool:
        """
        Open the camera connection.

        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            if self.camera_type == "csi":
                # Use GStreamer pipeline for CSI camera
                pipeline = self._get_csi_pipeline()
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                print(f"Opening CSI camera {self.camera_id} with GStreamer pipeline")
            else:
                # Use standard OpenCV for USB camera
                self.cap = cv2.VideoCapture(self.camera_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                print(f"Opening USB camera {self.camera_id}")

            if self.cap and self.cap.isOpened():
                self.is_opened = True
                print(f"✓ Camera opened successfully!")
                print(f"  Resolution: {self.width}x{self.height}")
                print(f"  Framerate: {self.framerate} fps")
                return True
            else:
                print(f"✗ Failed to open camera")
                return False

        except Exception as e:
            print(f"✗ Error opening camera: {e}")
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.

        Returns:
            Tuple of (success: bool, frame: np.ndarray or None)
        """
        if not self.is_opened or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("✓ Camera released")

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the current frame size.

        Returns:
            Tuple of (width, height)
        """
        return (self.width, self.height)


def main():
    """Test the camera reader."""
    import argparse

    parser = argparse.ArgumentParser(description="Test camera reader")
    parser.add_argument(
        "--type",
        choices=["csi", "usb"],
        default="csi",
        help="Camera type (csi or usb)"
    )
    parser.add_argument(
        "--id",
        type=int,
        default=0,
        help="Camera ID"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Frame width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Frame height"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second"
    )
    parser.add_argument(
        "--flip",
        type=int,
        default=0,
        help="Flip method for CSI camera (0-7)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Point-to-Explain Camera Reader Test")
    print("="*60)
    print("\nPress 'q' to quit, 's' to save a snapshot\n")

    # Create and open camera
    with CameraReader(
        camera_type=args.type,
        camera_id=args.id,
        width=args.width,
        height=args.height,
        framerate=args.fps,
        flip_method=args.flip
    ) as camera:

        if not camera.is_opened:
            print("Failed to open camera. Exiting.")
            return

        frame_count = 0
        snapshot_count = 0

        while True:
            ret, frame = camera.read()

            if not ret:
                print("Failed to read frame")
                break

            frame_count += 1

            # Display frame info
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Point-to-Explain Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"snapshot_{snapshot_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Saved {filename}")
                snapshot_count += 1

        cv2.destroyAllWindows()
        print(f"\nTotal frames captured: {frame_count}")


if __name__ == "__main__":
    main()

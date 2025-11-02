#!/usr/bin/env python3
"""
Unit tests for camera_reader module.
Tests camera initialization, frame capture, and resource management.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

# Add parent directory to path to import camera_reader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_reader import CameraReader


class TestCameraReader(unittest.TestCase):
    """Test suite for CameraReader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def tearDown(self):
        """Clean up after tests."""
        pass

    @patch('camera_reader.cv2.VideoCapture')
    def test_csi_camera_initialization(self, mock_video_capture):
        """Test CSI camera initialization with correct pipeline."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(camera_type="csi", camera_id=0)
        result = camera.open()

        self.assertTrue(result)
        self.assertTrue(camera.is_opened)
        self.assertIsNotNone(camera.cap)

        # Verify VideoCapture was called with GStreamer pipeline
        call_args = mock_video_capture.call_args
        pipeline = call_args[0][0]
        self.assertIn("nvarguscamerasrc", pipeline)
        self.assertIn("sensor-id=0", pipeline)
        self.assertIn("video/x-raw", pipeline)

    @patch('camera_reader.cv2.VideoCapture')
    def test_usb_camera_initialization(self, mock_video_capture):
        """Test USB camera initialization."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(camera_type="usb", camera_id=0)
        result = camera.open()

        self.assertTrue(result)
        self.assertTrue(camera.is_opened)

        # Verify VideoCapture was called with camera ID
        mock_video_capture.assert_called_once_with(0)

        # Verify camera properties were set
        mock_cap.set.assert_any_call(3, 1280)  # CAP_PROP_FRAME_WIDTH
        mock_cap.set.assert_any_call(4, 720)   # CAP_PROP_FRAME_HEIGHT
        mock_cap.set.assert_any_call(5, 30)    # CAP_PROP_FPS

    @patch('camera_reader.cv2.VideoCapture')
    def test_camera_open_failure(self, mock_video_capture):
        """Test handling of camera opening failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(camera_type="usb")
        result = camera.open()

        self.assertFalse(result)
        self.assertFalse(camera.is_opened)

    @patch('camera_reader.cv2.VideoCapture')
    def test_read_frame_success(self, mock_video_capture):
        """Test successful frame reading."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.mock_frame)
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(camera_type="usb")
        camera.open()

        ret, frame = camera.read()

        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))
        mock_cap.read.assert_called_once()

    @patch('camera_reader.cv2.VideoCapture')
    def test_read_frame_failure(self, mock_video_capture):
        """Test frame reading failure."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(camera_type="usb")
        camera.open()

        ret, frame = camera.read()

        self.assertFalse(ret)
        self.assertIsNone(frame)

    @patch('camera_reader.cv2.VideoCapture')
    def test_read_without_open(self, mock_video_capture):
        """Test reading frame without opening camera."""
        camera = CameraReader(camera_type="usb")

        ret, frame = camera.read()

        self.assertFalse(ret)
        self.assertIsNone(frame)

    @patch('camera_reader.cv2.VideoCapture')
    def test_camera_release(self, mock_video_capture):
        """Test camera resource release."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(camera_type="usb")
        camera.open()
        camera.release()

        mock_cap.release.assert_called_once()
        self.assertFalse(camera.is_opened)

    @patch('camera_reader.cv2.VideoCapture')
    def test_context_manager(self, mock_video_capture):
        """Test camera as context manager."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self.mock_frame)
        mock_video_capture.return_value = mock_cap

        with CameraReader(camera_type="usb") as camera:
            self.assertTrue(camera.is_opened)
            ret, frame = camera.read()
            self.assertTrue(ret)

        # Verify release was called
        mock_cap.release.assert_called_once()

    @patch('camera_reader.cv2.VideoCapture')
    def test_custom_resolution(self, mock_video_capture):
        """Test camera with custom resolution."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(
            camera_type="usb",
            width=1920,
            height=1080,
            framerate=60
        )
        camera.open()

        # Verify custom properties were set
        mock_cap.set.assert_any_call(3, 1920)  # Width
        mock_cap.set.assert_any_call(4, 1080)  # Height
        mock_cap.set.assert_any_call(5, 60)    # FPS

    @patch('camera_reader.cv2.VideoCapture')
    def test_csi_flip_method(self, mock_video_capture):
        """Test CSI camera with flip method."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(
            camera_type="csi",
            flip_method=2
        )
        camera.open()

        # Verify flip method in pipeline
        call_args = mock_video_capture.call_args
        pipeline = call_args[0][0]
        self.assertIn("flip-method=2", pipeline)

    def test_get_frame_size(self):
        """Test getting frame size."""
        camera = CameraReader(width=1920, height=1080)
        width, height = camera.get_frame_size()

        self.assertEqual(width, 1920)
        self.assertEqual(height, 1080)

    @patch('camera_reader.cv2.VideoCapture')
    def test_csi_pipeline_generation(self, mock_video_capture):
        """Test correct GStreamer pipeline generation for CSI camera."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        camera = CameraReader(
            camera_type="csi",
            camera_id=1,
            width=1280,
            height=720,
            framerate=30,
            flip_method=2
        )
        camera.open()

        call_args = mock_video_capture.call_args
        pipeline = call_args[0][0]

        # Check all required pipeline elements
        self.assertIn("nvarguscamerasrc sensor-id=1", pipeline)
        self.assertIn("width=(int)1280", pipeline)
        self.assertIn("height=(int)720", pipeline)
        self.assertIn("framerate=(fraction)30/1", pipeline)
        self.assertIn("flip-method=2", pipeline)
        self.assertIn("nvvidconv", pipeline)
        self.assertIn("videoconvert", pipeline)
        self.assertIn("appsink", pipeline)

    @patch('camera_reader.cv2.VideoCapture')
    def test_exception_handling_during_open(self, mock_video_capture):
        """Test exception handling during camera opening."""
        mock_video_capture.side_effect = Exception("Camera error")

        camera = CameraReader(camera_type="usb")
        result = camera.open()

        self.assertFalse(result)
        self.assertFalse(camera.is_opened)

    @patch('camera_reader.cv2.VideoCapture')
    def test_multiple_cameras(self, mock_video_capture):
        """Test opening multiple cameras with different IDs."""
        mock_cap1 = MagicMock()
        mock_cap1.isOpened.return_value = True

        mock_cap2 = MagicMock()
        mock_cap2.isOpened.return_value = True

        mock_video_capture.side_effect = [mock_cap1, mock_cap2]

        camera1 = CameraReader(camera_type="usb", camera_id=0)
        camera2 = CameraReader(camera_type="usb", camera_id=1)

        result1 = camera1.open()
        result2 = camera2.open()

        self.assertTrue(result1)
        self.assertTrue(result2)

        # Verify both cameras were opened with correct IDs
        calls = mock_video_capture.call_args_list
        self.assertEqual(calls[0][0][0], 0)
        self.assertEqual(calls[1][0][0], 1)


class TestCameraReaderIntegration(unittest.TestCase):
    """Integration tests for CameraReader."""

    @patch('camera_reader.cv2.VideoCapture')
    def test_typical_usage_workflow(self, mock_video_capture):
        """Test typical camera usage workflow."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)
        mock_video_capture.return_value = mock_cap

        # Typical workflow
        camera = CameraReader(camera_type="usb")

        # Open camera
        self.assertTrue(camera.open())

        # Read some frames
        for _ in range(10):
            ret, frame = camera.read()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)

        # Release camera
        camera.release()
        mock_cap.release.assert_called_once()

    @patch('camera_reader.cv2.VideoCapture')
    def test_context_manager_with_error(self, mock_video_capture):
        """Test context manager properly releases resources even with errors."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap

        try:
            with CameraReader(camera_type="usb") as camera:
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify camera was still released despite error
        mock_cap.release.assert_called_once()


def run_tests():
    """Run all tests with verbose output."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCameraReader))
    suite.addTests(loader.loadTestsFromTestCase(TestCameraReaderIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

"""
FaceMapper - contains the FaceMapper class which uses mediapipe to obtain real-time facial landmark data
"""
import time
import argparse
import cv2
from numpy import copy, ndarray
from typing import Optional
import sys
import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils, FaceLandmarkerResult, FaceLandmarksConnections,\
    RunningMode, FaceLandmarker, FaceLandmarkerOptions


def draw_landmarks_on_image(rgb_image: ndarray, detection_result: FaceLandmarkerResult):
    """
    function to draw facial landmark results onto an image
    :param rgb_image: the image to plot landmarks onto
    :param detection_result: the face landmark detection result
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = copy(rgb_image)
    # Loop through the detected faces to visualize
    for face_landmarks in face_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image


class FaceMapper:

    def __init__(self,
                 model_path: Optional[str] = 'vision/models/face_landmarker.task',
                 num_faces: int = 1,
                 min_detection_confidence: float = 0.8,
                 min_tracking_confidence: float = 0.8,
                 output_face_blendshapes: bool = False,
                 output_facial_transformation_matrixes: bool = False):
        """
        FaceMapper initialization
        :param model_path: Path to face_landmarker.task model file
        :param num_faces: Maximum number of faces to detect
        :param min_detection_confidence: Minimum confidence for detection
        :param min_tracking_confidence: Minimum confidence for tracking
        :param output_face_blendshapes: Whether to output face blendshapes
        :param output_facial_transformation_matrixes: Whether to output transformation matrices
        """
        print("Initializing FaceMapper...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
            )
        # create our landmarker
        base_options = BaseOptions(model_asset_path=model_path)
        options = FaceLandmarkerOptions(
            base_options= base_options,
            running_mode= RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_face_blendshapes,
            output_facial_transformation_matrixes=output_facial_transformation_matrixes
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.frame_counter = 0
        print(f"FaceMapper initialized successfully: Model: {model_path}, Max faces: {num_faces}")

    def process_image(self, image_input: str | ndarray, output_path: Optional[str] = None) -> ndarray:
        """
        Process a static image and detect facial landmarks
        :param image_input Path to input image or frame as ndarray
        :param output_path: Optional path to save annotated image
        :returns Annotated image
        """
        image = image_input
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not read image from {image_input}")
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Detect landmarks
        detection_result = self.landmarker.detect_for_video(mp_image, self.frame_counter)
        self.frame_counter += 1
        # Draw landmarks
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Saved annotated image to {output_path}")
        return annotated_image

    def process_webcam(self, camera_index: int = 0, window_name: str = "FaceMapper"):
        """
        function to process webcam feed in real-time
        args:
        :param camera_index: Camera device index (usually 0)
        :param window_name: Name of display window
        """
        webcam = cv2.VideoCapture(camera_index)
        if not webcam.isOpened():
            raise ValueError(f"Could not open camera with index {camera_index}")
        print("\nWebcam started")
        print("Controls:")
        print("  'q or ESC' - Quit")
        print("  's' - Save current frame")
        save_counter = 0
        # Recording loop
        while webcam.isOpened():
            success, frame = webcam.read()
            if not success:
                print("Failed to read frame")
                break
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            # Process and show frame
            annotated_frame = self.process_image(frame)
            cv2.imshow(window_name, annotated_frame)
            # Handle key inputs
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) :
                break
            elif key == ord('s'):
                filename = f"facemapper_capture_{time.time()}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Saved: {filename}")
                save_counter += 1
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        webcam.release()
        cv2.destroyAllWindows()

    def get_landmarks_array(self, image_input: str | ndarray) -> Optional[ndarray]:
        """
        Extract facial landmarks as numpy array
        :param image_input: Path to input image or a numpy array of the image frame
        :returns numpy array of facial landmarks or none if no face detected
        """
        image = image_input
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image from {image_input}")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self.landmarker.detect_for_video(mp_image, self.frame_counter)
        self.frame_counter += 1
        # Maps each landmark to a list of its xyz components
        return [[lm.x, lm.y,lm.z] for lm in detection_result.face_landmarks[0]] if detection_result.face_landmarks else None

    def cleanup(self):
        """
        Clean up function
        """
        self.landmarker.close()
        print("FaceMapper cleaned up")


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(description="FaceMapper - Facial Landmark Detection")
    parser.add_argument('--mode', choices=['image', 'webcam'], required=True,
                        help='Processing mode')
    parser.add_argument('--input', type=str, help='Input image path (for image mode)')
    parser.add_argument('--output', type=str, help='Output image path (for image mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (for webcam mode)')
    parser.add_argument('--model', type=str, help='Path to face_landmarker.task model file')
    args = parser.parse_args()
    try:
        mapper = FaceMapper(model_path=args.model)
        if args.mode == 'image':
            if not args.input:
                print("Error: --input required for image mode")
                sys.exit(1)
            print(f"\nProcessing: {args.input}")
            annotated = mapper.process_image(args.input, args.output)
            # Display
            cv2.imshow("FaceMapper", annotated)
            print("\nPress any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # Show landmark info
            landmarks = mapper.get_landmarks_array(args.input)
            if landmarks is not None:
                print(f"\n Detected {len(landmarks)} landmarks")
                print(f"Shape: {landmarks.shape}")
            else:
                print("\nNo face detected")

        elif args.mode == 'webcam':
            mapper.process_webcam(camera_index=args.camera)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if 'mapper' in locals():
            mapper.cleanup()


if __name__ == "__main__":
    main()
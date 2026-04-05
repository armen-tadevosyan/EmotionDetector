"""
FaceMapper manual testing program
"""
import time

from facemapper import FaceMapper
import cv2


def example_webcam():
    """
    Enables webcam with live facial landmark mapping
    """
    mapper = FaceMapper(num_faces=1)
    try:
        mapper.process_webcam(camera_index=0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mapper.cleanup()


def example_image():
    """
    Maps facial landmarks onto images of human faces
    """
    mapper = FaceMapper()
    try:
        image_path = input("\nEnter path to your image: ").strip()
        annotated = mapper.process_image(image_path, f'output_landmarks_{time.time()}.jpg')
        cv2.imshow("FaceMapper landmarks", annotated)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Saved output to: output_landmarks.jpg")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        mapper.cleanup()

def start():
    """
    Starts the test
    """
    print("\nAvailable Examples:")
    print("1. Webcam processing)")
    print("2. Image processing")
    choice = input("\nSelect example: ").strip()
    examples = {
        '1': example_webcam,
        '2': example_image,
    }
    example_func = examples.get(choice)
    if example_func:
        print()
        example_func()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
   start()
"""Pose analysis based on rotating image to align feet horizontally."""

import cv2
import os

from PIL.ImageShow import show

import rotated_posture_recongniser
from posture_controller import PostureController
from strategies import central_axis


# --- Helper function ---

def resize_image_for_display(image, max_width=1280, max_height=720):
    """Resize image to fit within display bounds."""
    h, w = image.shape[:2]

    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't enlarge if already smaller

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def main():
    """Image loading and processing loop.
    image_path = "test_images/andrea_front.jpg"

    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    controller = PostureController()
    processed_image = controller.process_image(image, strategy_name='central_axis')

    # recogniser = abstract_posture_recogniser.AbstractPostureRecogniser(image)
    # resized_image = resize_image_for_display(recogniser.annotated_image)

    cv2.imshow("Posture Analysis", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    controller = PostureController()
    image = cv2.imread("test_images/andrea_front.jpg")
    controller.process_image(image)

    controller.set_strategy("spiral_line")
    annotated_image = controller.annotate()

    cv2.imshow("Spiral Result", annotated_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

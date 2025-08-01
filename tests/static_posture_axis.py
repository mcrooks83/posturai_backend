"""Pose analysis based on rotating image to align feet horizontally."""

import cv2
import numpy as np
import os
import posture_recongniser


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
    """Image loading and processing loop."""
    image_path = "../test_images/andrea_front.jpg"

    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    recogniser = posture_recongniser.PostureRecogniser(image)
    resized_image = resize_image_for_display(recogniser.annotated_image)

    cv2.imshow("Posture Analysis", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

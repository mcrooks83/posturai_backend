"""Pose analysis based on rotating image to align feet horizontally."""

import cv2
import numpy as np
import os
import posture_recongniser


# --- Helper functions ---

def angles_between_axis_and_line(line, axis):
    """A tuple of joint angles."""
    line = np.array(line, dtype=float)
    axis = np.array(axis, dtype=float)

    line[1] = -line[1]
    axis[1] = -axis[1]

    dot_product = np.dot(line, axis)
    len_line = np.linalg.norm(line)
    len_axis = np.linalg.norm(axis)

    cos_angle = dot_product / (len_line * len_axis)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_1 = np.degrees(np.arccos(cos_angle))
    angle_2 = 180 - angle_1

    scalar = line[0] * axis[1] - line[1] * axis[0]

    if scalar > 0:
        left_angle = angle_1
        right_angle = angle_2
    else:
        left_angle = angle_2
        right_angle = angle_1
    return  left_angle, right_angle

def describe_tilt(line, axis, label):
    """Print the results."""
    a, b = angles_between_axis_and_line(line, axis)
    if abs(a - b) < 0.5:
        if label == "shoulders":
            return f"Your {label}s are pretty balanced"
        else:
            return f"Your {label} is pretty balanced"
    if a == max(a, b):
        if label == "shoulders":
            return f"Your right shoulder is elevated"
        else:
            return f"Your {label} is elevated on the right side"
    else:
        if label == "shoulders":
            return f"Your left shoulder is elevated"
        else:
            return f"Your {label} is elevated on the left side"

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
    image_path = "../test_images/orange_relaxed.jpg"  # Example: "./images/person1.jpg"

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

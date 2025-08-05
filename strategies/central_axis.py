"""Measure spiral line."""

from helper import vector_utils, image_annotation
from abstract_posture_recogniser import AbstractPostureRecogniser
from helper.landmark import Landmark


class CentralAxis(AbstractPostureRecogniser):

    def __init__(self, image, axis_finder=None):
        super().__init__(image)
        self.image = image
        self.analysis_result = self.analyze_posture()
        self.annotated_image = self.annotate(image)

    def analyze_posture(self):
        spine_status = ""
        knee_rotation_report = ""
        # HEAD
        # average all points on the left face and right face
        left_face, right_face = Landmark.get_face_side_centers()

        head_angle = vector_utils.get_angle_by_points(left_face, right_face)
        head_description = vector_utils.describe_tilt_with_angle("head", head_angle)

        # NECK
        # neck vector is reversed direction than the others (should make consistent)
        vec_neck = self.axis_finder.vec_neck
        vec_spine = self.axis_finder.vec_spine

        angle_between = vector_utils.find_angles_between(vec_spine, vec_neck)
        if angle_between[0] > 5:
            neck_flexion = "Your neck (cervical spine) is laterally flexed to the right"
        elif angle_between[0] < -5:
            neck_flexion = "Your neck (cervical spine) is laterally flexed to the left"
        else:
            neck_flexion = "Your neck (cervical spine) is alinged with the spine"

        shoulder_angle = self.axis_finder.shoulder_angle
        hip_angle = self.axis_finder.hip_angle
        shoulder_description = vector_utils.describe_tilt_with_angle("shoulders", shoulder_angle)
        pelvis_status = vector_utils.describe_tilt_with_angle("pelvis", hip_angle)

        # --- knee or femur internal rotation --- #
        left_dev = self.axis_finder.left_dev
        right_dev = self.axis_finder.right_dev
        threshold = 10  # pixels

        if abs(left_dev) < threshold and abs(right_dev) < threshold:
            knee_rotation_report = "Both knees show similar alignment"
        else:
            left_internal = left_dev < 0
            right_internal = right_dev < 0
            if left_internal and right_internal:
                if abs(left_dev) > abs(right_dev):
                    knee_rotation_report = "Left knee shows more internal rotation"
                elif abs(right_dev) > abs(left_dev):
                    knee_rotation_report = "Right knee shows more internal rotation"
                else:
                    knee_rotation_report = "Both knees show equal internal rotation"
            elif left_internal:
                knee_rotation_report = "Left knee shows internal rotation; right knee less so"
            elif right_internal:
                knee_rotation_report = "Right knee shows internal rotation; left knee less so"
            else:
                knee_rotation_report = "Neither knee shows internal rotation"

        # Pelvic rotation estimation
        left_hip_offset = self.axis_finder.left_hip_offset
        right_hip_offset = self.axis_finder.right_hip_offset

        if abs(left_hip_offset - right_hip_offset) < 20:
            pelvis_status += " and the pelvis is neutral"
        elif left_hip_offset > right_hip_offset:
            pelvis_status += " and the pelvis is rotated right (left hip forward)"
        else:
            pelvis_status += " and the pelvis is rotated left (right hip forward)"

        # spine direction relative to the base_mid
        vec_center_axis = (0, 1)  # Downward vertical in image coordinates, does assume that the image is straight
        spine_flex_angle = vector_utils.find_angles_between(vec_center_axis, vec_spine)
        # print(f"spine flex angle {spine_flex_angle}", flush=True)

        # Spinal lateral flexion
        if abs(spine_flex_angle) < 1:
            spine_status = "Your spine appears vertically aligned"
        elif 1 < spine_flex_angle < 5:
            spine_status = "Your spine is slightly flexed to the left"
        elif spine_flex_angle >= 5:
            spine_status = "Your spine is flexed to the left"
        elif -5 < spine_flex_angle < -1:
            spine_status = "Your spine is slightly flexed to the right"
        elif spine_flex_angle <= -5:
            spine_status = "Your spine is flexed to the right"

        # Spinal rotation
        left_shoulder_dist = self.axis_finder.left_shoulder_dist
        right_shoulder_dist = self.axis_finder.right_shoulder_dist
        rotation_diff = left_shoulder_dist - right_shoulder_dist

        if abs(rotation_diff) < 5:
            spine_status += " and is facing forward (neutral rotation)."
        elif rotation_diff > 0:
            spine_status += " and is rotated toward the left (right shoulder forward)."
        else:
            spine_status += " and is rotated toward the right (left shoulder forward)."

        # Compose text report
        report_text = f"""
    {head_description}
    {neck_flexion}
    {shoulder_description}
    {spine_status}
    {pelvis_status}
    {knee_rotation_report}
    """
        return report_text

    def annotate(self, landmarks):
        annotated_image = self.image.copy()

        image_annotation.draw_center_axis(self.image)
        image_annotation.draw_pose_landmarks(self.image, landmarks)

        return annotated_image

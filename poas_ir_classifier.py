from poas.ir.classifiers.fence_detector import FenceDetector
from poas.ir.classifiers.fumaroles_detector import FumarolesDetector
from poas.ir.classifiers.plume_detector import PlumeDetector
from poas.ir.classifiers.low_visibility_classifier import LowVisibilityClassifier
from poas.ir.classifiers.degraded_classifier import DegradedClassifier
import sys
import cv2

class IRImageClassifier:
    '''
        This classifier must be run on an IR image taken from the VPMI imaging system at the Poas Volcano.
        Changes to the assumed camera parameters will shift the results.
    '''
    def __init__(self):
        self.plume_detector = PlumeDetector()
        self.fence_detector = FenceDetector()
        self.fumaroles_detector = FumarolesDetector()
        self.low_visibility_classifier = LowVisibilityClassifier()
        self.degraded_classifier = DegradedClassifier()


    def classify_ir_img(self, img):
        assignments = set()

        has_plume = self.plume_detector.stage_1_has_plume(img) or self.plume_detector.stage_2_has_plume(img)
        is_degraded = self.degraded_classifier.is_degraded(img)
        has_fumaroles = self.fumaroles_detector.has_fumaroles(img, 75)
        has_fence = self.fence_detector.has_fence(img, 150, 220, 33000)

        if is_degraded:
            assignments.add("Degraded")
        if has_fumaroles:
            assignments.add("Fumaroles")
        if has_fence:
            assignments.add("Fence")

        if has_plume:
            assignments.add("Plume")
        else:
            if is_degraded:
                return assignments

            is_low_visibility = self.low_visibility_classifier.has_visible_components_in_plume_area(img)

            if is_low_visibility:
                assignments.add("Low visibility")
            else:
                if has_fence:
                    assignments.add("Obscured")
                else:
                    assignments.add("Cloud cover")
        
        return assignments


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python<3> poas_ir_classifier.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to read image at {image_path}")
        sys.exit(1)

    classifier = IRImageClassifier()
    assignments = classifier.classify_ir_img(img)
    print(f"Assignments for Image {image_path.split('/')[-1]}:", assignments)

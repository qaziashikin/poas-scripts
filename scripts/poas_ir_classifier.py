"""
This script runs the various classification stages for images at Poás volcano, Costa
Rica.

:author:
    Qazi T. Ashikin

"""

import argparse
import pathlib
import tomllib
import time

import cv2
import numpy as np
from classifiers.degraded_classifier import is_degraded
from classifiers.canny_feature_detector import detect_feature as canny_detect_feature
from classifiers.contour_feature_detector import (
    detect_feature as contour_detect_feature,
)


def classify(image: np.ndarray, config: dict) -> set[str]:
    """
    Run through the various classification stages.

    Parameters
    ----------
    image:
        The image to be classified.
    config:
        Information for each of the classification stages.

    Parameters
    ----------
     :
        The classification assignments.

    """

    assignments, feature_scores = set(), {}
    for classifier, classifier_config in config["classifiers"].items():
        if classifier_config["type"] == "canny_edge_sum":
            feature_score = canny_detect_feature(image, classifier_config)
            feature_detected = feature_score > classifier_config["threshold"]
        elif classifier_config["type"] == "contour_sum":
            feature_score = contour_detect_feature(image, classifier_config)
            feature_detected = feature_score > classifier_config["threshold"]

        if feature_detected:
            assignments.add(classifier.title())

        feature_scores[classifier.title()] = feature_score

    if is_degraded(image):
        assignments.add("Degraded")

    # Strong plume detection
    strong_fences_and_fumaroles = (
        feature_scores["Fence"] > 65000 and feature_scores["Fumaroles"] > 450
    )

    has_plume = strong_fences_and_fumaroles or "Plume" in assignments

    if has_plume:
        assignments.add("Plume")
        try:
            assignments.remove("Low_Visibility")
        except KeyError:
            pass
    else:
        if "Degraded" in assignments:
            return assignments

        if "Low_Visibility" not in assignments:
            if "Fence" in assignments:
                assignments.add("Obscured")
            else:
                assignments.add("Cloud cover")

    return assignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        help="Specify a path to image to be classified.",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Specify a path to a config file for the classifiers.",
        required=True,
    )
    args = parser.parse_args()

    image_path = pathlib.Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError

    ts = time.time()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image failed to load.")

    with pathlib.Path(args.config).open("rb") as f:
        config = tomllib.load(f)

    assignments = classify(image, config)
    print(time.time() - ts)
    print(f"Assignments for Image {image_path.name}", assignments)

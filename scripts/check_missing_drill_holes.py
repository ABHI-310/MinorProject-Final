"""Find candidate absent openings in an aligned board photo pair."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def find_absent_openings(reference, inspection):
    """Find dark green openings in the reference replaced by solid pads."""
    if reference.shape != inspection.shape:
        raise ValueError("The reference and inspection photos must have the same pixel dimensions.")
    grid_delta = cv2.absdiff(reference[::8, ::8], inspection[::8, ::8]).max(axis=2)
    agreement = float((grid_delta <= 20).mean())
    if agreement < 0.98:
        raise ValueError(f"The photos need closer alignment and lighting. Pixel agreement was {agreement:.1%}.")

    blue, green, red = cv2.split(reference.astype(np.int16))
    inspected_blue, inspected_green, inspected_red = cv2.split(inspection.astype(np.int16))
    reference_opening = (green - red >= 12) & (green - blue >= 12)
    color_change = np.maximum(inspected_blue - blue, inspected_red - red) >= 22
    brightness_gain = inspected_red + inspected_green + inspected_blue - red - green - blue >= 36
    inspection_pad = np.maximum(inspected_green - inspected_red, inspected_green - inspected_blue) <= 25
    mask = (reference_opening & color_change & brightness_gain & inspection_pad).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 25 <= area <= 6500:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        if min(width, height) / max(width, height) < 0.62 or 4 * np.pi * area / perimeter**2 < 0.55:
            continue
        candidates.append({"x": x + width / 2, "y": y + height / 2, "area_pixels": area})
    candidates.sort(key=lambda candidate: -candidate["area_pixels"])
    return agreement, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Aligned photo with the expected drill holes")
    parser.add_argument("inspection", type=Path, help="Photo to inspect")
    parser.add_argument("output", type=Path, help="Path for the JSON report")
    args = parser.parse_args()
    photos = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in (args.reference, args.inspection)]
    if any(photo is None for photo in photos):
        parser.error("Both inputs must be readable images.")
    try:
        agreement, candidates = find_absent_openings(*photos)
    except ValueError as error:
        parser.error(str(error))
    report = {
        "reference": str(args.reference),
        "inspection": str(args.inspection),
        "pixel_agreement": agreement,
        "candidate_absent_openings": candidates,
        "scope": "Exploratory check for aligned green-board images. Candidate locations require review.",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Candidate absent openings: {len(candidates)}")


if __name__ == "__main__":
    main()

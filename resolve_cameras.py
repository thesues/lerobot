#!/usr/bin/env python
"""Resolve OpenCV camera indices by device name (macOS).

OpenCV's AVFoundation backend (cap_avfoundation_mac.mm, v4.13) enumerates
[AVCaptureDevice devicesWithMediaType:Video] + [... :Muxed] and sorts the list
by uniqueID; VideoCapture(N) is the Nth device of that sorted list. This script
replicates the exact same enumeration via PyObjC and looks devices up by
localizedName, so the printed index always matches what cv2 will open. No
camera is opened in the process, so nothing can wedge.

uniqueID embeds the USB topology: replugging a camera into another port (or
the iPhone Continuity Camera appearing) still reshuffles indices — but this
resolver recomputes the mapping at every launch, so that no longer matters.

Usage:
    resolve_cameras.py front wrist     # prints one index per name, in order

Current rig: front = "USB Camera" (top-down view),
             wrist = "Innomaker-U20CAM-1080p-S1" (fisheye on the gripper).
"""

import sys

import objc

NAME_SUBSTRINGS = {"front": "USB Camera", "wrist": "Innomaker"}


def opencv_ordered_device_names() -> list[str]:
    objc.loadBundle(
        "AVFoundation", {}, bundle_path="/System/Library/Frameworks/AVFoundation.framework"
    )
    AVCaptureDevice = objc.lookUpClass("AVCaptureDevice")
    devices = list(AVCaptureDevice.devicesWithMediaType_("vide"))
    devices += list(AVCaptureDevice.devicesWithMediaType_("muxd"))
    devices.sort(key=lambda d: str(d.uniqueID()))  # same comparator as OpenCV
    return [str(d.localizedName()) for d in devices]


def main() -> None:
    queries = sys.argv[1:]
    if not queries or any(q not in NAME_SUBSTRINGS for q in queries):
        print(f"usage: resolve_cameras.py [{' | '.join(NAME_SUBSTRINGS)}] ...", file=sys.stderr)
        sys.exit(2)
    names = opencv_ordered_device_names()
    for query in queries:
        sub = NAME_SUBSTRINGS[query]
        hits = [i for i, name in enumerate(names) if sub.lower() in name.lower()]
        if len(hits) != 1:
            print(
                f"{query}: expected exactly one device matching {sub!r}, found {hits}; "
                f"devices in OpenCV index order: {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(hits[0])


main()

#!/usr/bin/env python3

"""
Calibration of LED marker positions using an AprilTag as common reference.

Pictures of LED markers are captured with single camera, from different positions
and orientation. An AprilTag is used to establish a common reference between the
snapshots. 3D marker positions are then determined using least-squares.

Each snapshot is expected to contain of two files:
- Framebuffer picture as recorded by the camera: *bmp bitmap (no compression)
- Frame and blob information: JSON file

The JSON file is expected to contain a dict with the following items:
    "frame"         Frame number/ID (int)
    "exposure_us"   Exposure time in [us] (int)
    "blobs"         List of blobs (list of dict)

    Each entry in the "blobs" list is a dict with the following items:
        "cxf"       Sub-pixel position of blob center, x-coordinate (float)
        "cyf"       Sub-pixel position of blob center, y-coordinate (float)
        "w"         Width of blob in [pixel] (int)
        "h"         Height of blob in [pixel] (int)
        "pixels"    Number of pixels in blob (int)


Camera data is loaded from a file defined in config.yaml file


(C) 2024, Felix Althaus

"""




import os
import sys
import tkinter as tk
from tkinter import filedialog
import json
from ruamel.yaml import YAML
import pupil_apriltags as apriltag
import numpy as np
import matplotlib.pyplot as plt
import cv2
import views





class Camera():
    """
    Simple basic monocular camera simulation.

    """
    def __init__(self, file):
        """
        Initialize camera object.

        Parameters:
            camera_matrix   3x3 Matrix of camera intrinsics

                                            |fx  0  cx |
                            camera_matrix = | 0 fy  cy |
                                            | 0  0   0 |
        Returns:
            Camera() object

        """

        yaml = YAML(typ='rt')
        with open(file, "r", encoding="utf-8") as cfgfile:
            camera_data = yaml.load(cfgfile)

        self.camera_matrix = np.array(camera_data["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(camera_data["dist_coeffs"], dtype=np.float64)
        self.fx = self.camera_matrix[0][0]
        self.fy = self.camera_matrix[1][1]
        self.cx = self.camera_matrix[0][2]
        self.cy = self.camera_matrix[1][2]


    def project(self, points):
        """
        Project points from 3D points to pixel coordinates.

        Parameters:
            points  3x1 3D vector (x,y,z) of point to be projected

        Returns:
            Projected points in pixel coordinates

        """
        points_projected = self.camera_matrix @ points
        return (points_projected / points_projected[2,:])


    def undistort_blobs(self, blobs):
        """
        Undistort blob ("cyf","cyf") positions

        Parameters:
            blobs   List of blobs
                        Each entry is a dict containing blob information

        Returns:
            Blob list with blob ("cxf","cyf") position undistorted

        Note: Blob width and height do not get undistorted, as both are supposed
        to be quite small, this can be ignored (and width and height are not
        further processed anyway)

        """
        src = np.zeros((len(blobs), 1, 2))
        dst = np.zeros((len(blobs), 1, 2))

        for i in range(src.shape[0]):
            src[i][0][0] = blobs[i]["cxf"]
            src[i][0][1] = blobs[i]["cyf"]

        # Termination criteria
        # (COUNT, 5, 0.01) seems to be the default i.a.w. the github opencv repository
        criteria = (cv2.TERM_CRITERIA_COUNT, 5, 0.01)
        #
        dst = cv2.undistortPointsIter(src, self.camera_matrix, self.dist_coeffs,
                                      None, self.camera_matrix, criteria=criteria)

        for i in range(dst.shape[0]):
            blobs[i]["cxf"] = dst[i][0][0]
            blobs[i]["cyf"] = dst[i][0][1]

        return blobs




if __name__ == "__main__":


    # read YML configuration file
    yaml = YAML(typ='rt')
    with open("config.yaml", "r", encoding="utf-8") as cfgfile:
        config = yaml.load(cfgfile)


    if config["testing"]:
        # if "testing" = True load test images from "test_images" list
        filenames = config["test_images"]
    else:
        # if "testing" = False, show file selection dialog for user to select
        root = tk.Tk()
        root.withdraw()	# hide tkinter root window (so only the file open dialog is shown)
        filenames = filedialog.askopenfilenames(title="Select Directory",
                                                filetypes=[("Bitmap", ".bmp")],
                                                initialdir="imgs")
        if not filenames:
            print("No files selected. Exiting.")
            sys.exit(0)


    print()


    camera = Camera(file=config["camera"]["file"])     # camera simulator

    # AprilTag outline at zero position (center defined at (0,0,0)
    tag0 = np.array([[-1/2,  1/2,  1/2, -1/2],
                     [ 1/2,  1/2, -1/2, -1/2],
                     [   0,    0,    0,    0]])*config["tag"]["size"]

    # set up AprilTag detector
    camera_params = [camera.fx, camera.fy, camera.cx, camera.cy]

    at_detector = apriltag.Detector(families="tag36h11")

    scene = views.SceneView()         # 3D scene view


    # Iterate over all pictures
    for filename in filenames:

        filename_noext = os.path.splitext(filename)[0]
        filename_basename = os.path.splitext(os.path.basename(filename))[0]

        print("File: '{:s}'".format(filename_basename))
        print()

        with open(filename_noext + ".json", "r", encoding='utf-8') as blobfile:
            framedata = json.load(blobfile)

        frame_id = framedata["frame"]
        exposure_us = framedata["exposure_us"]
        blobs = framedata["blobs"]
        blobs = camera.undistort_blobs(blobs)

        # TODO: filter for valid blobs (i.e. marker candidates)

        # We need to get the correct (=expected) number of markers/blobs
        if len(blobs) != config["expected_markers"]:
            raise ValueError("Unexpected number of blobs: "
                             "expected {:d}, got {:d}.".format(len(blobs),
                                                               config["expected_markers"]))

        # Sort blobs left-to-right to keep corresponding blobs in order
        # CAUTION: to establish correspondence, all LEDs are assumed
        #          to be seen from left to right!
        blobs.sort(key=lambda blob: blob["cxf"])

        blobvector = np.zeros((3,len(blobs)))
        # Show blob information
        print("    Frame #{:d}: {:d} us, {:d} blob(s)".format(frame_id, exposure_us, len(blobs)))
        for i, blob in enumerate(blobs):
            print("      {:d}: cxf={:.3f}, cyf={:.3f}, w={:d}, h={:d}".format(i,
                                                                              blob["cxf"],
                                                                              blob["cyf"],
                                                                              blob["w"],
                                                                              blob["h"]))
            # create normalized blob vector
            blobvector[:,i] = np.array([(blob["cxf"]-camera.cx)/camera.fx,
                                        (blob["cyf"]-camera.cy)/camera.fy,
                                                                      1.0])
            blobvector[:,i] = blobvector[:,i] / np.linalg.norm(blobvector[:,i])
        print()

        # Read in picture, undistort, and perform AprilTag detection
        img_uncorrected = cv2.imread(filename)
        img = cv2.undistort(img_uncorrected, camera.camera_matrix, camera.dist_coeffs)
        detected_tags = at_detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                           estimate_tag_pose=True,
                                           camera_params=camera_params,
                                           tag_size=config["tag"]["size"])

        # One AprilTag needs to be found (and only one)
        if len(detected_tags) == 1:
            tag = detected_tags[0]
            print("    AprilTag:")
            print("      Tag Family - ID:  {:s} - {:d}".format(tag.tag_family.decode(), tag.tag_id))
            print("      Corrected errors: {:d}".format(tag.hamming))
            print("      Decision margin:  {:.1f}".format(tag.decision_margin))
        else:
            raise ValueError(f"Expected 1 tag, found {len(detected_tags):d}")


        print()


        projview = views.ProjectedView(filename_basename, camera.camera_matrix)
        cameraview = views.CameraView(img, filename_basename, (camera.cx, camera.cy))
        projview.plot_detected_tag(tag)
        projview.plot_blobs(blobs)

        # Re-project back tag using estimated pose (pose_R, pose_t)
        tag_moved = (tag.pose_R @ tag0) + tag.pose_t

        # Re-project back center of tag using estimated pose
        tag_moved_center = (tag.pose_R @ [[0],[0],[0]]) + tag.pose_t

        projview.plot_projected_tag(camera.project(tag_moved), camera.project(tag_moved_center))

        cameraview.draw_tag(tag)
        cameraview.draw_blobs(blobs)


        scene.draw_tag(tag0, 'b-')

        print("    |pose_t| = {:.3f} m".format(np.linalg.norm(tag.pose_t)))
        tvec = tag.pose_R.T @ -tag.pose_t

        blobray0 = tag.pose_R.T @ blobvector[:,[0]]
        blobray1 = tag.pose_R.T @ blobvector[:,[1]]

        a0 = -tvec[2] / blobray0[2]
        a1 = -tvec[2] / blobray1[2]

        blob0 = tvec + a0*blobray0
        blob0_moved = (tag.pose_R @ blob0) + tag.pose_t

        blob1 = tvec + a1*blobray1
        blob1_moved = (tag.pose_R @ blob1) + tag.pose_t

        scene.draw_vect(np.hstack((np.zeros((3,1)), tvec)), 'k-')
        scene.draw_vect(np.hstack((tvec, blob0)), "m-")
        scene.draw_vect(np.hstack((tvec, blob1)), "m-")

        projview.plot_point(camera.project(blob0_moved), 'rx')
        projview.plot_point(camera.project(blob1_moved), 'rx')


        # estimated led position in 3D (z=0)
        dist = np.linalg.norm(blob0-blob1)
        print("    |blob[0] - blob[1]| = {:.0f} mm".format(dist*1000))



        cameraview.show()

        print()
        print()



    print("Close all windows to continue.")

    plt.show()

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()



# TODO: implement least-squares estimate of marker 3D positions
#
# Least Square Estimate of line crossing point (i.e. LED position)
#
# https://www.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space
# https://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines

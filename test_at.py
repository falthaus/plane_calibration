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


(C) 2024, Felix Althaus

"""




import os
import sys
import tkinter as tk
from tkinter import filedialog
import json
import pupil_apriltags as apriltag
import numpy as np
import matplotlib.pyplot as plt
import cv2
import views




# Camera intrinsic matrix [px]
camera_matrix_K = [[550,   0, 320],
                   [  0, 550, 240],
                   [  0,   0,   1]]

# Physical size of the AprilTag [m]
tag_size = 0.131

# AprilTag outline at zero position (center defined at (0,0,0)
tag0 = np.array([[-tag_size/2, tag_size/2,  tag_size/2, -tag_size/2],
                 [ tag_size/2, tag_size/2, -tag_size/2, -tag_size/2],
                 [          0,          0,           0,           0]])

N_EXPECTED_MARKERS = 2        # number of expected markers (valid blobs)



class Camera():
    """
    Simple basic monocular camera simulation.

    """
    def __init__(self, camera_matrix):
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
        self.C = camera_matrix


    def project(self, points):
        """
        Project points from 3D points to pixel coordinates.

        Parameters:
            points  3x1 3D vector (x,y,z) of point to be projected

        Returns:
            Projected points in pixel coordinates

        """
        points_projected = self.C @ points
        return (points_projected / points_projected[2,:])




if __name__ == "__main__":


    # quick and dirty way to use  fixed list of images for testing:
    # if '-t' argument is provided, files are loaded from 'snapshot.json' file
    if len(sys.argv) == 2:

        if sys.argv[1] == "-t":
            # Argument '-t' provided, load list of snapshots from JSON file
            with open("snapshots.json", "r", encoding="utf-8") as jsonfile:
                filenames = json.load(jsonfile)
        else:
            print("Unknown argument '{:s}'. Exiting".format(sys.argv[1]))
            sys.exit(-1)

    else:
        # no argument provided, show file selection dialog for user to select
        root = tk.Tk()
        root.withdraw()	# hide tkinter root window (so only the file open dialog is shown)
        filenames = filedialog.askopenfilenames(title="Select Directory",
                                                filetypes=[("Bitmap", ".bmp")],
                                                initialdir="imgs")
        if not filenames:
            print("No files selected. Exiting.")
            sys.exit(0)


    # set up AprilTag detector
    camera_params = [camera_matrix_K[0][0],
                     camera_matrix_K[1][1],
                     camera_matrix_K[0][2],
                     camera_matrix_K[1][2]]

    at_detector = apriltag.Detector(families="tag36h11")


    print()


    cam = Camera(camera_matrix_K)     # camera simulator

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
        blobvector = np.zeros((3,len(blobs)))

        # TODO: filter for valid blobs (i.e. marker candidates)

        # We need to get the correct (=expected) number of markers/blobs
        if len(blobs) != N_EXPECTED_MARKERS:
            raise ValueError("Unexpected number of blobs: "
                             "expected {:d}, got {:d}.".format(len(blobs), N_EXPECTED_MARKERS))


        # Sort blobs left-to-right to keep corresponding blobs in order
        # CAUTION: to establish correspondence, all LEDs are assumed
        #          to be seen from left to right!
        blobs.sort(key=lambda blob: blob["cxf"])

        # Show blob information
        print("    Frame #{:d}: {:d} us, {:d} blob(s)".format(frame_id, exposure_us, len(blobs)))
        for i, blob in enumerate(blobs):
            print("      {:d}: cxf={:.3f}, cyf={:.3f}, w={:d}, h={:d}".format(i,
                                                                              blob["cxf"],
                                                                              blob["cyf"],
                                                                              blob["w"],
                                                                              blob["h"]))
            # normalized blob vector
            blobvector[:,i] = np.array([(blob["cxf"]-camera_matrix_K[0][2])/camera_matrix_K[0][0],
                                        (blob["cyf"]-camera_matrix_K[1][2])/camera_matrix_K[1][1],
                                                                                             1.0])

            blobvector[:,i] = blobvector[:,i] / np.linalg.norm(blobvector[:,i])
        print()

        # Read in picture and perform AprilTag detection
        img = cv2.imread(filename)
        detected_tags = at_detector.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                           estimate_tag_pose=True,
                                           camera_params=camera_params,
                                           tag_size=tag_size)

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


        projview = views.ProjectedView(filename_basename, camera_matrix_K)
        cameraview = views.CameraView(img, filename_basename, camera_matrix_K)
        projview.plot_detected_tag(tag)
        projview.plot_blobs(blobs)

        # Re-project back tag using estimated pose (pose_R, pose_t)
        tag_moved = (tag.pose_R @ tag0) + tag.pose_t

        # Re-project back center of tag using estimated pose
        tag_moved_center = (tag.pose_R @ [[0],[0],[0]]) + tag.pose_t

        projview.plot_projected_tag(cam.project(tag_moved), cam.project(tag_moved_center))

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

        projview.plot_point(cam.project(blob0_moved), 'rx')
        projview.plot_point(cam.project(blob1_moved), 'rx')


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

"""
Helper functions to plot/display camera and 3D views.

- SceneView()
- CameraView()
- ProjectedView()

(C) 2024, Felix Althaus

"""




import matplotlib.pyplot as plt
import cv2
import numpy as np




class SceneView():
    """
    3D view with tag, camera and blobs

    """
    def  __init__(self, limits=1.2):
        """
        Initialize scene view (3D).

        Parameters:
            limits  Common x,y,z limits for 3D plot (optional)

        Returns:
            SceneView() object

        """
        self.fig = plt.figure()
        self.fig.canvas.manager.set_window_title("3D Scene View")
        self.ax = plt.axes(projection ='3d')
        self.ax.set_xlim3d(-limits, limits)
        self.ax.set_ylim3d(-limits, limits)
        self.ax.set_zlim3d(-limits, limits)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.invert_yaxis()  # to match camera coordinate system
        self.ax.invert_zaxis()  # to match camera coordinate system
        plt.xlabel("x")
        plt.ylabel("y")


    def draw_tag(self, tag_corners, fmt):
        """
        Draw AprilTag outline in 3D.

        Parameters:
            tag_corners     Four corners of AprilTag outline
            fmt             pyplot-compatible format string (i.e. color, style, marker)

        Returns:
            none
        """
        # AprilTag outline (append first point to end to get closed shape)
        self.ax.plot3D(np.append(tag_corners[0], tag_corners[0][0]),
                       np.append(tag_corners[1], tag_corners[1][0]),
                       np.append(tag_corners[2], tag_corners[2][0]), fmt, linewidth=0.7)
        # AprilTag center
        self.ax.plot3D(0,0,0, fmt[0]+"+")
        # AprilTag middle of bottom edge
        self.ax.plot3D(*((tag_corners[:,[0]] + tag_corners[:,[1]])/2), fmt[0]+"+")


    def draw_vect(self, vect, fmt):
        """
        Draw vector in 3D

        Parameters:
            vect    3x2 matrix: [3x1 start point, 3x1 end point]
            fmt     pyplot-compatible format string (i.e. color, style, marker)

        Returns:
            none
        """
        self.ax.plot3D([vect[0][0], vect[0][1]],
                       [vect[1][0], vect[1][1]],
                       [vect[2][0], vect[2][1]], fmt, linewidth=0.7)




class CameraView():
    """
    Original image with objects drawn on top of it.
    Uses OpenCV functions.

    """
    def __init__(self, image, name, camera_matrix):
        """
        Initialize

        Parameters:
            image           Original image (used as background to draw on)
            name            Name to show in window title.
            camera_matrix   3x3 Matrix of camera intrinsics

        Returns:
            CameraView() object

        """
        self.image = image
        self.name = name
        self.C = camera_matrix

        # draw camera crosshairs
        cv2.line(self.image, (self.C[0][2],              0),
                             (self.C[0][2], self.C[1][2]*2), color=(128,128,128))
        cv2.line(self.image, (             0, self.C[1][2]),
                             (self.C[0][2]*2, self.C[1][2]), color=(128,128,128))
        # Draw camera principal axis
        cv2.drawMarker(self.image, (self.C[0][2], self.C[1][2]), color=(0, 0, 0), markerSize=7)


    def draw_tag(self, tag_object):
        """
        Draw AprilTag outline and center.

        Parameters:
            tag_object  Object as returned by pupil_apriltags.detect()

        Returns:
            None

        """
        # Draw ArilTag corners
        corners = [tag_object.corners.reshape((-1,1,2)).astype(int)]
        cv2.polylines(self.image, corners, isClosed=True, color=(0, 0, 255))
        # Draw AprilTag center
        cv2.drawMarker(self.image, tag_object.center.astype(int), color=(0, 0, 255), markerSize=7)


    def draw_blobs(self, bloblist):
        """
        Draw blobs (detected markers)

        Parameters:
            bloblist    List of blobs
                        Each entry is a dict containing blob information

        Returns:
            None

        """
        for b in bloblist:
            # Blob bounding box
            cv2.rectangle(self.image, (b["x"], b["y"]),
                                      (b["x"] + b["w"], b["y"] + b["h"]),
                                       color=(255, 0, 255))
            # Blob center
            cv2.drawMarker(self.image, (round(b["cxf"]), round(b["cyf"])),
                                       color=(255, 0, 255), markerSize=7)


    def show(self):
        """
        Wrapper function for OpenCV imshow() to actually display the plot

        Parameters:
            None

        Returns:
            None

        """
        cv2.imshow(self.name, self.image)




class ProjectedView():
    """
    Synthesized 2D camera view. Has the same with and height as the actual
    camera, allowing direct comparison between picture and synthesized view.

    """
    def __init__(self, name, camera_matrix):
        """

        Parameters:
            name            Name to show in window title.
            camera_matrix   3x3 Matrix of camera intrinsics


        Returns:
            ProjectedView() object

        """
        # Fix plot to match exactly the height/width of the actual camera picture
        # https://stackoverflow.com/questions/71852403/matplotlib-how-to-set-plot-size-with-dpi-not-figure-size
        #
        margin = 0.06
        px = 1/plt.rcParams['figure.dpi']
        self.fig, self.ax = plt.subplots(figsize=(640*px/(1-2*margin), 480*px/(1-2*margin)))
        self.fig.subplots_adjust(left=margin, right=1-margin, bottom=margin, top=1-margin)
        self.fig.canvas.manager.set_window_title(name)
        self.ax.set_xlim(0, 640)
        self.ax.set_ylim(0, 480)
        self.ax.invert_yaxis()
        self.C = camera_matrix

        # make window not re-sizable (works only when using "TkAgg" backend)
        # https://stackoverflow.com/questions/33881554/how-do-i-lock-matplotlib-window-resizing
        bck = plt.get_backend()
        mng = plt.get_current_fig_manager()
        if bck == "TkAgg":
            mng.window.resizable(False, False)

        # Plot camera crosshairs and optical axis
        self.ax.plot([self.C[0][2], self.C[0][2]  ],
                     [           0, self.C[1][2]*2], 'gray', linewidth=0.7)

        self.ax.plot([           0, self.C[0][2]*2],
                     [self.C[1][2], self.C[1][2]  ], 'gray', linewidth=0.7)

        self.ax.plot(self.C[0][2], self.C[1][2], 'k+')


    def plot_detected_tag(self, tag_object):
        """
        Plot AprilTag outline as detected by pupil_apriltags.

        Parameters:
            tag_object  Object as returned by pupil_apriltags.detect()

        Returns:
            None

        """
        # FIXME: combine with plot_projected_tag()

        # Plot estimated tag outline
        self.ax.plot(np.append(tag_object.corners.T[0], tag_object.corners.T[0,0]),
                     np.append(tag_object.corners.T[1], tag_object.corners.T[1,0]),
                     'r-', linewidth=0.5)

        # Plot estimated tag center
        self.ax.plot(tag_object.center[0], tag_object.center[1], 'r+')


    def plot_projected_tag(self, tag_outline, center):
        """
        Plot projected tag.

        Parameters:
            tag_outline     Tag outline as a 2x4 matrix of pixel coordinates
            center          Tag center as a 2x1 vector of pixel coordinates

        Returns:
            None

        """
        # FIXME: combine with plot_detected_tag()

        # Plot reprojected tag outline
        self.ax.plot(np.append(tag_outline[0], tag_outline[0,0]),
                     np.append(tag_outline[1], tag_outline[1,0]), 'b-', linewidth=0.5)

        # Plot reprojected tag center
        self.ax.plot(center[0], center[1], 'b+')

        # tag middle of bottom edge
        self.ax.plot(*((tag_outline[0:2,[0]] + tag_outline[0:2,[1]])/2), 'b+')


    def plot_blobs(self, bloblist):
        """
        Plot a blob with center and bounding box.

        Parameters:
            bloblist    List of blobs
                        Each entry is a dict containing blob information
        Returns:
            None

        """
        for idx, b in enumerate(bloblist):

            # plot blob enter and bounding box
            self.ax.plot(round(b["cxf"]), round(b["cyf"]), 'm+')

            self.ax.plot([ b["x"], b["x"]+b["w"], b["x"]+b["w"], b["x"], b["x"] ],
                         [ b["y"], b["y"], b["y"]+b["h"], b["y"]+b["h"], b["y"] ],
                         "m-", linewidth=0.7)

            # plot blob index
            self.ax.text(b["x"]-1, b["y"]+b["h"]+1, f"{idx:d}",
                         horizontalalignment="right", verticalalignment="top",
                         fontsize="x-small", fontweight="ultralight")


    def plot_point(self, point, fmt):
        """
        Plot a single point.

        Parameters:
            point   2x1 vector of point in pixel coordinates
            fmt     pyplot-compatible format string (i.e. color, style, marker)

        Returns:
            None

        """
        self.ax.plot(point[0], point[1], fmt, linewidth=0.7)

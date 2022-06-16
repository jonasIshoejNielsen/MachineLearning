from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import math
import numpy as np

import matplotlib
matplotlib.use("tkagg")

def to_homogeneous(points):
    # Convert column vectors to row vectors
    if len(points.shape) == 1:
        points = points.reshape((*points.shape, 1))
    return np.vstack((points, np.ones((1, points.shape[1]))))


def to_euclidean(points):
    return points[:3]

def draw_3d_cube(vertices, M = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])):
    """This function draw a 3D cube based on its vertices."""
    rows, cols = vertices.shape
    if rows != 3 or cols != 8:
        return
    
    vertices =  to_euclidean(np.dot(M, to_homogeneous(vertices)))

    global max_x, max_y, max_z
    max_values = 1.1 * np.amax(abs(vertices.T), axis=0)
    max_x = max_values[0] if max_values[0] > max_x else max_x
    max_y = max_values[1] if max_values[1] > max_y else max_y
    max_z = max_values[2] if max_values[2] > max_z else max_z

    ax.cla()
    ax.grid(b=True, which="major")
    ax.scatter3D([0], [0], [0], s=20, c="black", marker="x")

    # list of sides' polygons of figure
    verts = [[vertices[:, 0], vertices[:, 1], vertices[:, 2], vertices[:, 3]],
             [vertices[:, 4], vertices[:, 5], vertices[:, 6], vertices[:, 7]],
             [vertices[:, 0], vertices[:, 1], vertices[:, 5], vertices[:, 4]],
             [vertices[:, 2], vertices[:, 3], vertices[:, 7], vertices[:, 6]],
             [vertices[:, 1], vertices[:, 2], vertices[:, 6], vertices[:, 5]],
             [vertices[:, 4], vertices[:, 7], vertices[:, 3], vertices[:, 0]],
             [vertices[:, 2], vertices[:, 3], vertices[:, 7], vertices[:, 6]]]
    
    #verts = np.dot(M, verts)
    ax.add_collection3d(
        Poly3DCollection(verts, facecolors="white", linewidths=1, alpha=0.5))
    ax.add_collection3d(
        Line3DCollection(verts, colors="black", linewidths=0.2, linestyles=":"))

    for i in range(0, cols):
        xs = vertices[0, i]
        ys = vertices[1, i]
        zs = vertices[2, i]
        ax.scatter3D(xs, ys, zs, s=50, marker="o")

    ax.set_xlabel("X-Axis")
    ax.set_xlim([-max_x, max_x])
    ax.set_ylabel("Y-Axis")
    ax.set_ylim([-max_y, max_y])
    ax.set_zlabel("Z-Axis")
    ax.set_zlim([-max_z, max_z])


def update_3d_cube(val):
    """
    This function will be performed when the user changes the 3D theta sliders.
    """
    theta_x = x_slider_theta.val
    theta_y = y_slider_theta.val
    theta_z = z_slider_theta.val

    # <Exercise 4.4 (3)>
    M= np.dot(np.dot(get_3d_rotation_matrix(np.radians(theta_x), 0), get_3d_rotation_matrix(np.radians(theta_y), 1)), get_3d_rotation_matrix(np.radians(theta_z), 2))
    draw_3d_cube(verticesCube, M)
    fig.canvas.draw_idle()


def get_2d_rotation_matrix(theta):
    """
    This function return a rotation matrix given an input theta angle in
    radians.
    """

    # Exercise 4.4 (2)
    return np.eye(2, dtype=np.float64)  # Change this


def get_3d_rotation_matrix(theta, axis=0):
    """
    This function return a rotation matrix given an input theta angle in
    radians.
    """

    # <Exercise 4.4 (1)>
    Rx = np.array([[1,0,0,0], [0, np.cos(theta), -np.sin(theta), 0], [0, np.sin(theta), np.cos(theta), 0], [0,0,0,1]])
    Ry = np.array([[np.cos(theta),0,np.sin(theta),0], [0, 1, 0, 0], [-np.sin(theta), 0, np.cos(theta), 0], [0,0,0,1]])
    Rz = np.array([[np.cos(theta),-np.sin(theta),0,0], [0, np.sin(theta), np.cos(theta), 0], [0, 0, 1, 0], [0,0,0,1]])
    if axis == 0 : 
        return Rx
    if axis == 1 : 
        return Ry
    if axis == 2 : 
        return Rz
    return np.eye(3, dtype=np.float64)  # Change this

def reset_sliders(t):
    x_slider_theta.reset()
    y_slider_theta.reset()
    z_slider_theta.reset()


fig = plt.figure()
plt.subplots_adjust(left=0.13, bottom=0.21, right=0.90, top=1.00)
ax = fig.add_subplot(111, projection="3d")

max_x = 0
max_y = 0
max_z = 0

# Create a 3D cube based on 8 vertices.
verticesCube = np.array([[1, 1, 11, 11, 1, 1, 11, 11],
                         [1, 11, 11,  1,  1, 11, 11,  1],
                         [1,  1,  1,  1, 11, 11, 11, 11]])


draw_3d_cube(verticesCube)

axcolor = "lightgoldenrodyellow"
x_slider_ax = plt.axes([0.2, 0.144, 0.65, 0.03], facecolor=axcolor)
x_slider_theta = Slider(x_slider_ax, "Theta (X)", 0.0, 360.0, valinit=0)
x_slider_theta.on_changed(update_3d_cube)

y_slider_ax = plt.axes([0.2, 0.107, 0.65, 0.03], facecolor=axcolor)
y_slider_theta = Slider(y_slider_ax, "Theta (Y)", 0.0, 360.0, valinit=0)
y_slider_theta.on_changed(update_3d_cube)

z_slider_ax = plt.axes([0.2, 0.07, 0.65, 0.03], facecolor=axcolor)
z_slider_theta = Slider(z_slider_ax, "Theta (Z)", 0.0, 360.0, valinit=0)
z_slider_theta.on_changed(update_3d_cube)

reset_ax = plt.axes([0.75, 0.025, 0.1, 0.04])
reset_button = Button(reset_ax, "Reset", color=axcolor, hovercolor="0.975")
reset_button.on_clicked(reset_sliders)

plt.show()

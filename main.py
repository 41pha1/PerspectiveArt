import numpy as np
import cv2 as cv
from perspective import lookat, perspective

position = np.array([0., 0., 0.])
lookingAt = np.array([1., 1., 1.])
up = np.array([0., 1., 0.])
fov = 90

cube_corners = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
], dtype=np.float64)

cube_faces = np.array([
    [0, 1, 3, 2],
    [0, 1, 5, 4],
    [0, 2, 6, 4],
])

target_image = cv.imread("target.png")
w, h = target_image.shape[:2]

def projectVoxel(voxel, viewProjectionMatrix):

    screen_verts = []

    for vert in voxel:
        vert = np.append(vert, 1.)
        projected = np.matmul(viewProjectionMatrix, vert).A1
        projected /= projected[3]
        screen_verts.append(projected[:2])

    return screen_verts

viewMatrix = lookat(position, lookingAt, up)
projectionMatrix = perspective(fov, w/h, 0.001, 1000)
viewProjectionMatrix = np.matmul(projectionMatrix, viewMatrix)

voxel = projectVoxel(cube_corners + np.array([1, 1, 1]), viewProjectionMatrix)

for face in cube_faces:
    # index into voxel to get face verts
    face_verts = []
    for vert in face:
        face_verts.append(voxel[vert])
    
    # draw face
    face_verts = np.array(face_verts)

    face_verts[:,0] = face_verts[:,0] * w/2 + h/2
    face_verts[:,1] = face_verts[:,1] * h/2 + w/2
    face_verts = np.round(face_verts).astype(np.int32)

    for i in range(len(face_verts)):
        cv.line(target_image, tuple(face_verts[i]), tuple(face_verts[(i+1)%len(face_verts)]), (255, 0, 0), 5)

cv.imwrite("output.png", target_image)

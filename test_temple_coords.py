import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2


# 1. Load the two temple images and the points from data/some_corresp.npz
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']

im1 = cv2.cvtColor(cv2.imread('../data/im1.png'), cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(cv2.imread('../data/im2.png'), cv2.COLOR_BGR2RGB)

# 2. Run eight_point to compute F
M = max(im1.shape[0], im1.shape[1])
F = sub.eight_point(pts1, pts2, M)


# 3. Load points in image 1 from data/temple_coords.npz
data = np.load('../data/temple_coords.npz')
temple_pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
temple_pts2 = sub.epipolar_correspondences(im1, im2, F, temple_pts1)

# 5. Compute the camera projection matrix P1
data = np.load('../data/intrinsics.npz')
K1 = data['K1']
K2 = data['K2']
E = sub.essential_matrix(F, K1, K2)

print(E)
# print('Essential Matrix:\n', E)
P1 = K1 @ np.eye(3,4)

# 6. Use camera2 to get 4 camera projection matrices P2
candidates = hlp.camera2(E)
P2s = [K2 @ candidates[:,:,i] for i in range(4)]



# 7. Run triangulate using the projection matrices
pts_3ds_and_errors = []
num_positive_zs = []
reprojection_errors = []
for i in range(4):
    pts_3d, reprojection_error = sub.triangulate(P1, temple_pts1, P2s[i], temple_pts2)
    num_positive_zs.append((pts_3d[:,2] > 0).sum())
    pts_3ds_and_errors.append((pts_3d, reprojection_error))
    

pts_3ds_and_errors_some = []
num_positive_zs_some = []
reprojection_errors_some = []
for i in range(4):
    pts_3d_some, reprojection_error_some = sub.triangulate(P1, pts1, P2s[i], pts2)
    num_positive_zs_some.append((pts_3d_some[:,2] > 0).sum())
    pts_3ds_and_errors_some.append((pts_3d_some, reprojection_error_some))


# 8. Figure out the correct P2 by choosing the one with the most points in front of both cameras
correct_index = np.argmax(num_positive_zs)
P2 = P2s[correct_index]
pts_3d, reproj_error = pts_3ds_and_errors[correct_index]

correct_index_some = np.argmax(num_positive_zs_some)
P2 = P2s[correct_index_some]
pts_3d_some, reproj_error_some = pts_3ds_and_errors_some[correct_index_some]

print(f'Reprojection error Temple: {reproj_error}')

print(f'Reprojection error Some: {reproj_error_some}')

# 9. Scatter plot the correct 3D points [No changes required here]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz [No changes required here]
E1 = np.eye(3,4)
E2 = candidates[:,:,correct_index]
R1, t1 = E1[:,:3], E1[:,[3]]
R2, t2 = E2[:,:3], E2[:,[3]]
np.savez('../data/extrinsics.npz', R1=R1, R2=R2, t1=t1, t2=t2)

import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def find_match(img1, img2):

    """
    Use OpenCV SIFT to extract keypoints and match between two views
    using k-nearest neighbor search. The matches will be filtered using
    the ratio test and bidirectional consistency check
    Input:
       img1: gray-scale image
       img2: gray-scale image
    Output:
       pts1: n x 2 matrices that specify the correspondence
       pts2: n x 2 matrices that specify the correspondence
    """

    # Code exactly the same as in HW2

    # SIFT feature extraction
    # kp: the coordinates of the keypoints
    # des: the descriptors of the keypoints
    # img1: template
    sift1 = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift1.detectAndCompute(img1, None)
    # img2: target
    sift2 = cv2.xfeatures2d.SIFT_create()
    kp2, des2 = sift2.detectAndCompute(img2, None)
    # Use two sets of descriptors from the template and target,
    # find the matches using nearest neighbor with the ratio test.
    # Nearest neighbor: find the top two descriptors of target
    # that is closest to each descriptor of the template.
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(des2)
    distances, indices = nbrs.kneighbors(des1, n_neighbors=2)
    # x1, x2: n x 2 matrices that specify the correspondence
    x1 = np.empty((0, 2))
    x2 = np.empty((0, 2))
    # Ratio test:
    for i in range(len(kp1)):
        if (distances[i, 0] / distances[i, 1]) < 0.7:
            x1 = np.append(x1, np.array([kp1[i].pt]), axis=0)  # pt: coordinates of the keypoints
            x2 = np.append(x2, np.array([kp2[indices[i, 0]].pt]), axis=0)

    return x1, x2


def compute_F(pts1, pts2):
    """
    Given matches, robustly compute a fundamental matrix by the 8-point algorithm
    with RANSAC.
    Input:
        pts1: n x 2 matrices that specify the correspondence
        pts2: n x 2 matrices that specify the correspondence
    Output:
        F: 3 x 3 fundamental matrix
    """

    ransac_thr = 0.01
    ransac_iter = 1000
    n_kp = pts1.shape[0]

    # track the maximum number of inliers of RANSAC
    max_num_inliers = 0
    for i in range(ransac_iter):
        samples = np.random.choice(n_kp, 8, replace=False)
        pts1_ransa = pts1[samples, :]
        pts2_ransa = pts2[samples, :]

        # Compute F matrix using 8-point
        A = np.zeros((8, 9))
        for j in range(8):
            A[j, :] = np.outer(np.append(pts2_ransa[j], 1), np.append(pts1_ransa[j], 1)).reshape(-1)
        u, d, vt = np.linalg.svd(A)
        v = np.transpose(vt)
        F_bf = v[:,-1].reshape((3,3))
        ## SVD clean-up to make rank of 2
        u, d, vt = np.linalg.svd(F_bf)
        d[2] = 0
        F_cleanup = np.matmul(u*d, vt)

        # compute loss
        A = np.zeros((n_kp, 9))
        for j in range(n_kp):
            A[j, :] = np.outer(np.append(pts2[j], 1), np.append(pts1[j], 1)).reshape(-1)
        L = np.matmul(A, F_cleanup.reshape(-1))
        L = abs(L)

        # compare the loss with the threshold to decide the inliers
        num_inliers = np.sum(L < ransac_thr)

        # compare with the maximum number of inliers so far
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            F = F_cleanup

    return F


def triangulation(P1, P2, pts1, pts2):

    """
    Given camera pose and correspondences, triangulate to reconstruct 3D points
    Input:
        P1: camera projection matrices, 3 x 4
        P2: camera projection matrices, 3 x 4
        pts1: n x 2 matrices that specify the correspondence
        pts2: n x 2 matrices that specify the correspondence
    Output:
        pts3d: n x 3, each row specifies the 3D reconstructed point
    """

    n_kp = pts1.shape[0]
    pts3D = np.zeros((n_kp, 3))
    for i in range(n_kp):
        pts1_skew_symmteric = np.array([[0, -1, pts1[i, 1]],
                                       [1, 0, -pts1[i, 0]],
                                       [-pts1[i, 1], pts1[i, 0], 0]])
        pts2_skew_symmteric = np.array([[0, -1, pts2[i, 1]],
                                       [1, 0, -pts2[i, 0]],
                                       [-pts2[i, 1], pts2[i, 0], 0]])
        pts1_P1 = np.matmul(pts1_skew_symmteric, P1)
        pts2_P2 = np.matmul(pts2_skew_symmteric, P2)

        # use SVD to find the null space
        A = np.concatenate((pts1_P1, pts2_P2), axis=0)
        u, d, vt = np.linalg.svd(A)
        v = np.transpose(vt)
        v_0 = v[:, 3]
        # normalize to make the last component 1
        v_0 = v_0 / v_0[3]
        pts3D[i, :] = v_0[0:3]

    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):

    """
    Given four configurations of relative camera pose and reconstructed points,
    find the best camera pose by verifying through 3D triangulation.
    Input:
        Rs: list of rotation matrices
        Cs: list of camera centers
        pts3Ds: list of 3D reconstructed points
    Output:
        R: best camera rotation
        C: best camera center
        pts3D: best 3D reconstructed points
    """
    n_cam = len(Rs)

    max_valid = 0
    for i in range(n_cam):
        r3 = Rs[i][2, :]
        cam = Cs[i]
        X = pts3Ds[i]
        # Cheirality condition
        cond = np.matmul(X - np.transpose(cam), r3)
        n_valid = np.sum(cond > 0)
        print(n_valid)

        # find the camera with the maximum number of valid points
        if n_valid > max_valid:
            max_valid = n_valid
            R = Rs[i]
            C = Cs[i]
            pts3D = pts3Ds[i]

    return R, C, pts3D


def compute_rectification(K, R, C):

    """
    Given the disambiguated camera pose, implement dense stereo matching between
    two views based on dense SIFT
    Input:
        K: intrinsic parameter
        R: relative camera rotation
        C: relative camera center
    Output:
        H1: homographies that rectify the left images
        H2: homographies that rectify the right images
    """

    # Compute the rectification rotation matrix
    C = C.reshape(-1)
    rx = C / np.linalg.norm(C)
    rz_tilde = np.array([0, 0, 1])
    rz = rz_tilde - np.matmul(rz_tilde, np.transpose(rx)) * rx
    rz = rz / np.linalg.norm(rz)
    ry = np.cross(rz, rx)
    R_rect = np.array([rx, ry, rz])

    # Compute the left and right homographies
    K_R_rect = np.matmul(K, R_rect)
    H1 = np.matmul(K_R_rect, np.linalg.inv(K))
    H2 = np.matmul(np.matmul(K_R_rect, np.transpose(R)), np.linalg.inv(K))

    return H1, H2


def dense_match(img1, img2):
    """
    Compute the dense matches across all pixels.
    Input:
        img1: grey-scale rectified image
        img2: grey-scale rectified image
    Output:
        disparity: H x W, disparity map, where H and W
                   are the image height and width
    """
    H, W = img1.shape
    size = 3

    sift = cv2.xfeatures2d.SIFT_create()

    #  Get the keypoint for each pixel
    keypoints = []
    for j in range(H):
        for i in range(W):
            keypoint = cv2.KeyPoint(i, j, size)
            keypoints.append(keypoint)

    #  Compute sift descriptor at each keypoint
    keypoints1, dense_feature1 = sift.compute(img1, keypoints)
    dense_feature1 = dense_feature1.reshape((H, W, 128))
    keypoints2, dense_feature2 = sift.compute(img2, keypoints)
    dense_feature2 = dense_feature2.reshape((H, W, 128))

    #  Compute the disparity map
    # disparity = np.zeros((H, W))
    # for j in range(H):
    #     for i in range(W):
    #         df1 = dense_feature1[j, i, :]
    #         min_dist = np.infty
    #         for k in range(W - i):
    #             df2 = dense_feature2[j, i + k, :]
    #             dist = np.linalg.norm(df1 - df2)
    #             if dist < min_dist:
    #                 best_match = k
    #                 min_dist = dist
    #         disparity[j, i] = best_match

    disparity = np.zeros((H, W))
    for j in range(H):
        for i in range(W):
            if img1[j, i] == 0:
                disparity[j, i] = 0
            else:
                df1 = dense_feature1[j, i, :]
                min_dist = np.infty
                for k in range(i + 1):
                    df2 = dense_feature2[j, i - k, :]
                    dist = np.linalg.norm(df1 - df2)
                    if dist < min_dist:
                        best_match = k
                        min_dist = dist
                disparity[j, i] = best_match

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})

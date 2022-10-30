import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

# # Draw the detected keypoints
# # draw a circle with size of keypoint and also its orientation
# img = cv2.imread('./Hyun_Soo_target1.jpg')
# gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints_target1.jpg',img)


def find_match(img1, img2):
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
            x1 = np.append(x1, np.array([ kp1[i].pt ]), axis=0) # pt: coordinates of the keypoints
            x2 = np.append(x2, np.array([ kp2[indices[i, 0]].pt ]), axis=0)

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # Compute an affine transform using SIFT matches filtered by RANSAC
    n_kp = x1.shape[0]

    # track the maximum number of inliers of RANSAC
    max_num_inliers = 0
    for i in range(ransac_iter):
        # DoF of Affine transformation: 6
        # 3 points required to solve affine transformation
        samples = np.random.choice(n_kp, 3, replace=False)
        x1_ransa = x1[samples, :]
        x2_ransa = x2[samples, :]
        a = np.array([
            [x1_ransa[0, 0], x1_ransa[0, 1], 1, 0, 0, 0],
            [0, 0, 0, x1_ransa[0, 0], x1_ransa[0, 1], 1],
            [x1_ransa[1, 0], x1_ransa[1, 1], 1, 0, 0, 0],
            [0, 0, 0, x1_ransa[1, 0], x1_ransa[1, 1], 1],
            [x1_ransa[2, 0], x1_ransa[2, 1], 1, 0, 0, 0],
            [0, 0, 0, x1_ransa[2, 0], x1_ransa[2, 1], 1]
        ])
        b = x2_ransa.reshape(6)
        if np.linalg.matrix_rank(a) < 6:
            print('Linear system singular')
        affine_trans = np.linalg.solve(a, b)
        affine_trans = np.append(affine_trans, np.array([0, 0, 1])).reshape((3, 3))

        # affine transformation of x1 as the estimation of x2
        x1_1 = np.hstack((x1, np.ones((n_kp, 1))))
        x2_est = np.matmul(x1_1, affine_trans.T)[:, 0:2]

        # compute the distances between the estimated x2 and true x2
        diff = np.sqrt(np.sum((x2 - x2_est)**2, axis=-1))
        # compare the distances with the threshold to decide the inliers
        num_inliers = np.sum(diff < ransac_thr)

        # compare with the maximum number of inliers so far
        if num_inliers > max_num_inliers:
            max_num_inliers = num_inliers
            A = affine_trans

    return A



def warp_image(img, A, output_size):
    h, w = output_size
    target_points = np.zeros((h, w, 2))
    for i in range(h):
        for j in range(w):
            target_point = np.matmul(A, np.array([j, i, 1]))
            # use interpn function from scipy.interpolate for bilinear interpolation
            # target_point: (y, x)
            target_points[i, j, :] = np.array([target_point[1], target_point[0]])
    # points: The points defining the regular grid in 2 dimensions
    points = (np.arange(img.shape[0]), np.arange(img.shape[1]))
    img_warped = interpolate.interpn(points, img, target_points)

    return img_warped

# Copied from homework 1
def get_differential_filter():
    # filter_x: 3 x 3 filter that differentiate along x using sobel filter
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # filter_y: 3 x 3 filter that differentiate along y using solbel filter
    filter_y = np.transpose(filter_x)
    return filter_x, filter_y

# Copied from homework 1
def filter_image(im, filter):
    # pad zeros on the boundary on the input image
    # to get the same size filtered image
    im_zero_pad = np.pad(im, 1)
    # get the size of im
    m, n = im.shape
    # generate im_filtered with the same size of im
    im_filtered = np.zeros((m, n))
    # image filtering
    for i in range(m):
        for j in range(n):
            # get the sub-matrix of im for filtering
            im_sub = im_zero_pad[i:i + 3, j:j + 3]
            # get the (i,j) value of im_filtered
            im_filtered[i, j] = np.sum(im_sub * filter)
    return im_filtered






def align_image(template, target, A):
    # line 2: Compute the gradient of template image
    filter_u, filter_v = get_differential_filter()
    I_du = filter_image(template, filter_u)
    I_dv = filter_image(template, filter_v)

    steepest_descent_images = np.zeros((template.shape[0], template.shape[1], 6))
    hessian = np.zeros((6, 6))
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            # line 3: Compute the Jacobian dW/dp at (x;0)
            Jacob = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
            # line 4: Compute the steepest descent images
            steepest_descent_images[i, j, :] = np.matmul(np.array([I_du[i, j], I_dv[i, j]]), Jacob)
            # line 5: Compute the 6 x 6 Hessian
            hessian = hessian + np.matmul(steepest_descent_images[i, j, :].reshape((6, 1)), steepest_descent_images[i, j, :].reshape((1, 6)))

    hessian_inv = np.linalg.inv(hessian)
    epsilon = 1
    errors = np.array([])
    iterations = 0
    max_iter = 1000
    # initialize A_refined with A
    A_refined = A
    # get initial p from A
    p = np.array([[A[0, 0] - 1, A[0, 1], A[0, 2], A[1, 0], A[1, 1] - 1, A[1, 2]]])
    while np.sqrt(np.sum(p**2)) > epsilon:
        iterations = iterations + 1
        if iterations > max_iter:
            break
        print('iteration {}'.format(iterations))
        # line 7: Warp the target to the template domain
        target_warped = warp_image(target, A_refined, template.shape)
        # line 8: Compute the error image
        I_err = target_warped - template
        error = np.sqrt(np.sum(I_err ** 2))
        errors = np.append(errors, error)


        # line 9
        F = np.zeros(6)
        for i in range(6):
            F[i] = np.sum(steepest_descent_images[:, :, i] * I_err)
        # line 10
        dp = np.matmul(hessian_inv, F.reshape((6, 1)))
        # line 11
        A_dp = np.array([[dp[0, 0]+1, dp[1, 0], dp[2, 0]], [dp[3, 0], dp[4, 0]+1, dp[5, 0]], [0, 0, 1]])
        A_refined = np.matmul(A_refined, np.linalg.inv(A_dp))
        # update p
        p = np.array([[A_refined[0, 0] - 1, A_refined[0, 1], A_refined[0, 2], A_refined[1, 0], A_refined[1, 1] - 1, A_refined[1, 2]]])

    return A_refined, errors


def track_multi_frames(template, img_list):
    ransac_thr = 10
    ransac_iter = 1000
    # Initialize the affine transform using the feature based alignment
    x1, x2 = find_match(template, img_list[0])
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    # track over frames using the inverse compositional image aligntment
    A_list = []
    for i in range(len(img_list)):
        print('Target image {}'.format(i+1))
        target = img_list[i]
        A, errors = align_image(template, target, A)
        template = warp_image(target, A, template.shape)
        template = np.array(template, dtype='uint8')
        A_list.append(A)
    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    np.random.seed(5561)
    ransac_thr = 10
    ransac_iter = 1000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    # # Visualize the boundary of the transformed template, Figure (3a) in report
    # boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
    #                                   [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
    # plt.imshow(target_list[0], cmap='gray')
    # plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'r')
    # plt.axis('off')
    # plt.show()


    # # Visualize the warped image, Figure (3c) in report
    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    # # Visualize the error map, Figure (3d) in report
    # err_img = np.abs(img_warped - template)
    # plt.imshow(err_img, cmap='jet')
    # plt.axis('off')
    # plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    # # Visualize the optimized affine transform using inverse compositional image alignment, Figure (4c) in report
    # boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
    #                                   [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A_refined[:2, :].T
    # plt.imshow(target_list[0], cmap='gray')
    # plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'r')
    # plt.axis('off')
    # plt.show()
    # # Visualize the comparison between initial warp and opt. warp, Figure (4d)
    # # Draw the error plot, Figure (4e)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    np.random.seed(5561)
    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)



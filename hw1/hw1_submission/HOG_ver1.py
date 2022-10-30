import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # filter_x: 3 x 3 filter that differentiate along x using sobel filter
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # filter_y: 3 x 3 filter that differentiate along y using solbel filter
    filter_y = np.transpose(filter_x)
    return filter_x, filter_y


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


def get_gradient(im_dx, im_dy):
    # grad_mag: magnitude of the gradient image
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    # grad_angle: orientation of the gradient image
    m, n = im_dx.shape
    grad_angle = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            grad_angle[i, j] = np.arctan2(im_dy[i, j], im_dx[i, j])
    # the range of the angle should be [0, pi)
    # for theta < 0, set theta == theta + pi
    grad_angle[grad_angle < 0] = grad_angle[grad_angle < 0] + np.pi
    # for theta = pi, set theta == 0
    grad_angle[grad_angle == np.pi] = 0
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # m x n: size of the original image
    m, n = grad_mag.shape
    # M: number of cells along y axes
    M = np.floor(m / cell_size).astype(int)
    # N: number of cells along x axes
    N = np.floor(n / cell_size).astype(int)
    # ori_histo: 3D tensor with size M x N x 6
    ori_histo = np.zeros((M, N, 6))
    for i in range(M):
        for j in range(N):
            cell_angle = grad_angle[i * cell_size: (i+1) * cell_size, j * cell_size: (j+1) * cell_size]
            cell_mag = grad_mag[i * cell_size: (i+1) * cell_size, j * cell_size: (j+1) * cell_size]
            # convert from cell_angle in [0,pi) to cell_degree in [0, 180)
            cell_degree = np.degrees(cell_angle)
            # bin 1: [165, 180) U [0, 15)
            ori_histo[i, j, 0] = np.sum(cell_mag[np.logical_or(cell_degree < 15, cell_degree >= 165)])
            # bin 2: [15, 45)
            ori_histo[i, j, 1] = np.sum(cell_mag[np.logical_and(cell_degree >= 15, cell_degree < 45)])
            # bin 3: [45, 75)
            ori_histo[i, j, 2] = np.sum(cell_mag[np.logical_and(cell_degree >= 45, cell_degree < 75)])
            # bin 4: [75, 105)
            ori_histo[i, j, 3] = np.sum(cell_mag[np.logical_and(cell_degree >= 75, cell_degree < 105)])
            # bin 5: [105, 135)
            ori_histo[i, j, 4] = np.sum(cell_mag[np.logical_and(cell_degree >= 105, cell_degree < 135)])
            # bin 6: [135, 165)
            ori_histo[i, j, 5] = np.sum(cell_mag[np.logical_and(cell_degree >= 135, cell_degree < 165)])
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    M, N, bins = ori_histo.shape
    eps = 0.001
    M_norm = M - block_size + 1
    N_norm = N - block_size + 1
    # ori_histo_normalized is the normalized histogram of size
    # (M-(block_size-1)) x (N-(block_size-1)) x (bins x block_size^2)
    ori_histo_normalized = np.zeros((M_norm, N_norm, bins * block_size * block_size))
    for i in range(M_norm):
        for j in range(N_norm):
            # the denominator for the normalized histogram
            denom = np.sqrt(np.sum(ori_histo[i: i + block_size, j: j + block_size, :]**2) + eps**2)
            # normalize the histogram and concatenate to form one long vector
            ori_histo_normalized[i, j, :] = (ori_histo[i: i + block_size, j: j + block_size, :] / denom).reshape(-1)
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0

    # get differential filters
    filter_x, filter_y = get_differential_filter()
    # Compute the filtered images
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    # compute the magnitude and angle of the gradient
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # build histogram of oriented gradients for each cell,
    # typical cell_size is 8
    ori_histo = build_histogram(grad_mag, grad_angle, 8)
    # Build the descriptor of all blocks with normalization
    # use block_size = 2
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)
    hog = ori_histo_normalized.reshape(-1)

    # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size ** 2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized ** 2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi / num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size * num_cell_w: cell_size],
                                 np.r_[cell_size: cell_size * num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    target_h, target_w = I_target.shape
    template_h, template_w = I_template.shape
    template_area = template_h * template_w

    # HOG descriptor of template
    hog_template = extract_hog(I_template)
    # normalized HOG descriptors of template
    hog_template_centered = hog_template - np.mean(hog_template)
    hog_template_norm = np.sqrt(np.sum(hog_template_centered ** 2))

    # (x,y) is the left-top corner coordinate of the bounding box
    # w_max: largest x-coordinate
    # h_max: largest y_coordinate
    h_max = target_h - template_h + 1
    w_max = target_w - template_w + 1

    # bounding_boxes: n x 3 array that describes the n detected bounding boxes
    boxes = np.zeros((w_max * h_max, 3))
    for i in range(h_max):
        print(h_max - i)
        for j in range(w_max):
            box = I_target[i: i+template_h, j: j+template_h]
            hog_box = extract_hog(box)
            hog_box_centered = hog_box - np.mean(hog_box)
            hog_box_norm = np.sqrt(np.sum(hog_box_centered ** 2))
            # ncc: normalized cross-correlation score
            ncc = np.sum(hog_box_centered * hog_template_centered) / (hog_box_norm * hog_template_norm)
            boxes[i * w_max + j, :] = np.array([j, i, ncc])

    # First thresholding on NCC score
    threshold = 0.48
    thresholding_boxes = boxes[boxes[:, 2] > threshold, :]

    # Then do Non-Maximum Suppression
    # we extract their NCC scores
    nccs = thresholding_boxes[: ,2]
    # sort the ncc scores and record the sorting indices
    sort_indx = np.argsort(nccs)


    # initialise an empty array for indices of non-maximum suppression bounding boxes
    keep = []
    while np.shape(sort_indx)[0] > 0:
        # get the index of the bounding box with the maximum ncc score
        max_indx = sort_indx[-1]
        # save it in the keep list
        keep.append(max_indx)

        # extract their coordinates ordered by their ncc scores
        x1 = thresholding_boxes[sort_indx, 0]
        y1 = thresholding_boxes[sort_indx, 1]
        # get the coordinates of the bounding box with the maximum score
        max_x1 = x1[-1]
        max_y1 = y1[-1]

        # calculate the widths and heights of the overlapping box between the maximum box
        # and the other boxes
        overlap_w = np.maximum(template_w - np.abs(max_x1 - x1), 0)
        overlap_h = np.maximum(template_h - np.abs(max_y1 - y1), 0)
        # calculate the area of the overlapping parts
        overlap_area = overlap_w * overlap_h
        # calculate the IoU
        IoU = overlap_area / (2 * template_area - overlap_area)
        # keep the boxes with IoU less than 0.5
        sort_indx = sort_indx[IoU < 0.5]

    bounding_boxes = thresholding_boxes[keep, :]
    return bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):
    hh, ww, cc = I_target.shape

    fimg = I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii, 0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1 < 0:
            x1 = 0
        if x1 > ww - 1:
            x1 = ww - 1
        if x2 < 0:
            x2 = 0
        if x2 > ww - 1:
            x2 = ww - 1
        if y1 < 0:
            y1 = 0
        if y1 > hh - 1:
            y1 = hh - 1
        if y2 < 0:
            y2 = 0
        if y2 > hh - 1:
            y2 = hh - 1
        fimg = cv2.rectangle(fimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f" % bounding_boxes[ii, 2], (int(x1) + 1, int(y1) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    # visualize the HOG descriptor
    im = im.astype('float') / 255.0
    visualize_hog(im, hog, 8, 2)

    I_target = cv2.imread('target.png', 0)
    # MxN image

    I_template = cv2.imread('template.png', 0)
    # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c = cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # this is visualization code.

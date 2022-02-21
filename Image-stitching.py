import cv2
import numpy as np
np.random.seed(7)
import random
random.seed(7)


def build_matrix_equation(final_left_keys, final_right_keys):

    a = np.zeros(shape=(2*len(final_left_keys), 9))
    for i in range(0, len(final_left_keys)):
        xn = np.array([final_left_keys[i].pt[0], final_left_keys[i].pt[1], 1, 0, 0, 0, -final_right_keys[i].pt[0]*final_left_keys[i].pt[0],
                       -final_right_keys[i].pt[0]*final_left_keys[i].pt[1], -final_right_keys[i].pt[0]])
        yn = np.array([0, 0, 0, final_left_keys[i].pt[0], final_left_keys[i].pt[1], 1, -final_right_keys[i].pt[1]*final_left_keys[i].pt[0],
                       -final_right_keys[i].pt[1]*final_left_keys[i].pt[1], -final_right_keys[i].pt[1]])
        a[2*i] = xn
        a[(2*i)+1] = yn
    return a

def solution(left_img, right_img):
    # Converting images into gray scales
    left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # Key-point detection and description
    key_points_left, descriptors_left = sift.detectAndCompute(left, None)
    key_points_right, descriptors_right = sift.detectAndCompute(right, None)

    # dictionary for storing top 2 matches
    distances_from_left = {}
    indices_r = [i for i in range(0, descriptors_right.shape[0])]
    # Key-point matching
    for i in range(descriptors_left.shape[0]):
        distance = np.sqrt(((descriptors_right - descriptors_left[i])**2).sum(axis=1))
        all_dist = dict(zip(indices_r, distance))
        vals = {k: v for k, v in sorted(all_dist.items(), key=lambda item: item[1])}
        distances_from_left[i] = (list(vals.keys())[:2])
    # ratio testing
    final_matches = {}
    for i in range(len(distances_from_left)):
        distance1 = np.sqrt(((descriptors_left[i] - descriptors_right[distances_from_left[i][0]])**2).sum())
        distance2 = np.sqrt(((descriptors_left[i] - descriptors_right[distances_from_left[i][1]])**2).sum())
        if distance1 < 0.75*distance2:
            final_matches[i] = distances_from_left[i][0]

    final_left_keys = [key_points_left[i] for i in final_matches.keys()]
    final_right_keys = [key_points_right[i] for i in final_matches.values()]

    # RANSAC algorithm
    total_in_liners_left = []
    total_in_liners_right = []
    for k in range(5000):
        left_four_ind = random.sample(final_matches.keys(), 4)
        right_four_ind = [final_matches[k] for k in left_four_ind]
        left_four = [key_points_left[i] for i in left_four_ind]
        right_four = [key_points_right[i] for i in right_four_ind]
        a = build_matrix_equation(left_four, right_four)
        U, s, VT = np.linalg.svd(a)
        h_cap = VT[-1]
        h_cap = h_cap.reshape((3, 3))
        left_remaining_ind = [i for i in final_matches.keys() if i not in left_four_ind]
        right_remaining_ind = [final_matches[k] for k in left_remaining_ind]
        left_remaining = [key_points_left[i] for i in left_remaining_ind]
        right_remaining = [key_points_right[i] for i in right_remaining_ind]
        in_liners_left = []
        in_liners_right = []
        for i in range(len(final_matches)-4):
            pi = np.array([[right_remaining[i].pt[0]], [right_remaining[i].pt[1]], [1]])
            prod = np.dot(h_cap, pi)
            norm_prod_x, norm_prod_y = prod[0][0]/prod[2][0], prod[1][0]/prod[2][0]
            l2 = (left_remaining[i].pt[0]-norm_prod_x)**2 + (left_remaining[i].pt[1]-norm_prod_y)
            if l2 < 5:
                in_liners_left.append(left_remaining[i])
                in_liners_right.append(right_remaining[i])
        if len(in_liners_left) > len(total_in_liners_left):
            total_in_liners_left = in_liners_left
            total_in_liners_right = in_liners_right

    a = build_matrix_equation(total_in_liners_left, total_in_liners_right)
    U, s, VT = np.linalg.svd(a)
    h_cap = VT[-1]
    h_cap = h_cap.reshape((3, 3))
    h = h_cap/h_cap[2][2]

    h1, w1 = left.shape
    h2, w2 = right.shape
    right_pts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    left_pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    left_t = cv2.perspectiveTransform(left_pts, h)
    pts = np.concatenate((right_pts, left_t), axis=0)
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
    trans_dist = [-x_min, -y_min]
    h_t = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]])
    output = cv2.warpPerspective(left_img, h_t.dot(h), ((x_max-x_min), (y_max-y_min)))
    tp, btm, lt, rt = -y_min, y_max-h2, -x_min, x_max-w2
    right_img = cv2.copyMakeBorder(right_img, tp, btm, lt, rt, borderType=cv2.BORDER_CONSTANT)
    h_, w_ = output.shape[:2]
    black = np.zeros(3)
    for i in range(h_):
        for j in range(w_):
            img1 = output[i, j, :]
            img2 = right_img[i, j, :]
            if not np.array_equal(img1, black) and np.array_equal(img2, black):
                output[i, j, :] = img1
            elif np.array_equal(img1, black) and not np.array_equal(img2, black):
                output[i, j, :] = img2
            elif not np.array_equal(img1, black) and not np.array_equal(img2, black):
                output[i, j, :] = img2
            else:
                pass
    result_img = output.astype(np.uint8)
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
# Computer Vision Image Stitching
 
Stitched left and right images together to create a panorama image. Made use of SIFT point detector in order to find key points and descriptors on both the given images. Using KNN, calculated the best matching pixels with the help of 2-norm distances between every key point in the left image to every other key point in the right image. To find the best matches, I made use of ratio testing with threshold = 0.7. Got a list of all the in-liners using the RANSAC algorithm, which in turn helped in the calculation of the estimated homography matrix. Used perspective transformation, and warped the left image into the right image and aligned the warped image. Then stitched the image to create a perfect panorama image.
Made use of the openCV library for basic keypoint extraction and implemented all the other algorithms from scratch.

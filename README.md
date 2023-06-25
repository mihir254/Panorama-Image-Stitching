# Panorama - Image Stitching

This repository contains the implementation of a panorama - image stitching from two input images: a left image and a right image.

## Overview

1. **Key Point Identification**: The project starts with the identification of key points in both images using a technique called SIFT (Scale-Invariant Feature Transform).

2. **Key Point Matching**: Each key point from the left image is compared with those in the right image using a method called KNN (K-Nearest Neighbors). The best matches are then filtered using a ratio test.

3. **Match Refinement**: The matches are further refined using a process called RANSAC (Random Sample Consensus) to discard potential errors or outliers.

4. **Transformation Matrix Computation**: Using these refined matches, a transformation matrix is computed, which provides the alignment details for the left image with respect to the right image.

5. **Image Warping and Stitching**: The left image is then warped based on the transformation matrix, aligned with the right image, and stitched together to create the final panoramic image.

This project makes use of the OpenCV library for initial key point detection, with the rest of the algorithms developed from scratch.

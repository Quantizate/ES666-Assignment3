import glob
import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm  # For progress bars
from src.Implementation.blending import Blender


class PanaromaStitcher:
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    i = 0
    homographyMatrix = []

    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        print(f"Loading images from {path}...")
        all_images = sorted(glob.glob(path + os.sep + "*"))
        print(f"Found {len(all_images)} Images for stitching")

        self.images = []
        for i, img in enumerate(all_images):
            img = cv2.imread(img)
            height, width = img.shape[:2]
            new_width = 400
            new_height = int((new_width / width) * height)
            img = cv2.resize(img, (new_width, new_height))
            self.images.append(img)

        for i, img in enumerate(self.images):
            if img is not None:
                print(f"Image {i} shape: {img.shape}")
            else:
                print(f"Image {i} could not be loaded.")

        self.homographyMatrix = [None] * len(self.images)

        self.stitch_all()

        b = Blender()

        panaroma = cv2.imread("output_images/warped_image0.jpg")

        length = len(self.images)
        for j in range(1, length):
            img = cv2.imread(f"output_images/warped_image{j}.jpg")
            # mask_new = (img > 0).astype(np.uint8)
            # mask_base = (img_cpy > 0).astype(np.uint8)
            # mask_combined = mask_new & mask_base

            panaroma = b.blend(panaroma, img)

        _, mask = b.getMask(panaroma)
        yi, xi = np.where(mask)
        minx = np.min(xi)
        maxx = np.max(xi)
        miny = np.min(yi)
        maxy = np.max(yi)
        panaroma = panaroma[miny:maxy, minx:maxx]

        # keypoints, descriptors = self.detect_and_compute(images)
        # matches = self.match_features(descriptors)

        # # self.drawMatches(images[0], images[1], keypoints[0], keypoints[1], matches[0], [1]*len(matches[0]))
        # # self.drawMatches(images[1], images[2], keypoints[1], keypoints[2], matches[1], [1]*len(matches[0]))

        # homographies = self.estimate_homographies(keypoints, matches)
        # stitched_image = self.stitch_images(images, homographies)

        # return stitched_image, homographies

        return panaroma, self.homographyMatrix

    def detectFeatures(self, image1, image2, ratio=0.75, maxMatches=500):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(image1, None)
        kp2, des2 = self.sift.detectAndCompute(image2, None)

        raw_matches = self.bf.knnMatch(des1, des2, k=2)

        matches = []
        keypoints = []
        for m, n in raw_matches:
            if m.distance < ratio * n.distance:
                matches.append(m)
                keypoints.append((kp1[m.queryIdx].pt, kp2[m.trainIdx].pt))

        matches = np.array(matches[:maxMatches])
        keypoints = np.array(keypoints[:maxMatches])

        return keypoints, matches

    def compute_homography(self, src_pts, dst_pts):
        A = []
        for j in range(len(src_pts)):
            x, y = src_pts[j][0]
            xp, yp = dst_pts[j][0]
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]  # Normalize so that H[2,2] is 1
        return H

    def ransac_homography(self, src_pts, dst_pts, num_iterations=10000, threshold=5.0):
        max_inliers = 0
        best_H = None
        for _ in range(num_iterations):
            # Randomly sample 4 points
            indices = np.random.choice(len(src_pts), 4, replace=False)
            src_sample = src_pts[indices]
            dst_sample = dst_pts[indices]

            # Compute homography using the sampled points
            H = self.compute_homography(src_sample, dst_sample)

            # Count inliers
            inliers = 0
            for j in range(len(src_pts)):
                src_pt = np.append(src_pts[j][0], 1)
                dst_pt = np.append(dst_pts[j][0], 1)
                projected_pt = np.dot(H, src_pt)
                projected_pt /= projected_pt[2]
                error = np.linalg.norm(projected_pt - dst_pt)
                if error < threshold:
                    inliers += 1

            # Update best homography if current one has more inliers
            if inliers > max_inliers:
                max_inliers = inliers
                best_H = H

        return best_H

    def pointMapping(self, point, M):
        new_point = np.dot(M, np.array([point[0], point[1], 1]))
        new_point_norm = new_point / new_point[2]
        new_point_norm = new_point_norm[:2].astype(np.int32)
        return new_point_norm

    def wrapImage(self, img, M, final_img, offset=(0, 0)):
        h, w, _ = img.shape

        # Forward transform of image endpoints
        endpts = np.int32([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])  # (x,y)
        endpts_t = np.int32([self.pointMapping(pt, M) for pt in endpts])
        endpts_t[:, 0] = (
            np.clip(endpts_t[:, 0] + offset[0], 0, final_img.shape[1]) - offset[0]
        )
        endpts_t[:, 1] = (
            np.clip(endpts_t[:, 1] + offset[1], 0, final_img.shape[0]) - offset[1]
        )

        # Find the image bounds range of the warped image
        maxX, minX = np.max(endpts_t[:, 0]), np.min(endpts_t[:, 0])
        maxY, minY = np.max(endpts_t[:, 1]), np.min(endpts_t[:, 1])
        rangeX = maxX - minX
        rangeY = maxY - minY
        # print("RangeX: ", rangeX, "\t RangeY: ", rangeY)
        # Find all pixel coordinates inside the warped image bounds
        coords = np.zeros((rangeX * rangeY, 2), dtype=np.int32)
        for i in range(rangeX):
            for j in range(rangeY):
                coords[(i * rangeY + j), :] = [minX + i, minY + j]

        m = np.linalg.inv(M)

        # Inverse transform of all other pixel coordinates using einsum for faster computation
        coords_homogeneous = np.hstack([coords, np.ones((coords.shape[0], 1))])
        coords_t_homogeneous = np.einsum("ij,kj->ki", m, coords_homogeneous)
        epsilon = 1e-10
        coords_t_homogeneous[:, 2] = np.where(
            coords_t_homogeneous[:, 2] == 0, epsilon, coords_t_homogeneous[:, 2]
        )
        coords_t = (
            coords_t_homogeneous[:, :2] / coords_t_homogeneous[:, 2][:, np.newaxis]
        ).astype(np.int32)
        coords_t = np.nan_to_num(coords_t, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.int32
        )

        # coords_t = np.zeros_like(coords)
        # for i in range(len(coords)):
        #     coords_t[i] = self.pointTransform(coords[i], m)

        # Filter transformed coordinates which are outside input image boundary
        bounded_indices = np.where(
            (0 <= coords_t[:, 0])
            & (coords_t[:, 0] < w)
            & (0 <= coords_t[:, 1])
            & (coords_t[:, 1] < h)
        )

        coords = coords[bounded_indices]
        coords_t = coords_t[bounded_indices]

        # Add warped image to the destination image
        valid_indices = np.where(
            (0 <= coords[:, 1] + offset[1])
            & (coords[:, 1] + offset[1] < final_img.shape[0])
            & (0 <= coords[:, 0] + offset[0])
            & (coords[:, 0] + offset[0] < final_img.shape[1])
        )
        coords = coords[valid_indices]
        coords_t = coords_t[valid_indices]
        final_img[coords[:, 1] + offset[1], coords[:, 0] + offset[0]] = img[
            coords_t[:, 1], coords_t[:, 0]
        ]

        output_dir = "output_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"warped_image{self.i}.jpg")
        cv2.imwrite(output_path, final_img)
        print(f"Warped image saved at {output_path}")

        # Clear the destination image
        final_img.fill(0)

    def stitch_img(self, img1, img2, dst_img, offset, homography=None):
        keypoints, matches = self.detectFeatures(img1, img2, ratio=0.5)

        src_points = np.float32([i for (_, i) in keypoints]).reshape(-1, 1, 2)
        dst_points = np.float32([i for (i, _) in keypoints]).reshape(-1, 1, 2)

        H = self.ransac_homography(src_points, dst_points)
        self.homographyMatrix[self.i] = H

        H = np.dot(homography, H)
        self.wrapImage(img2, H, dst_img, offset)

        return H

    def stitch_all(self):
        final_img = np.zeros((2500, 8000, 3), dtype=np.uint8)
        offset = (3000, 800)

        mid_index = len(self.images) // 2

        homography = np.eye(3)
        for i in range(mid_index, 0, -1):
            self.i = i - 1
            homography = self.stitch_img(
                self.images[i], self.images[i - 1], final_img, offset, homography
            )

        homography = np.eye(3)
        self.i = mid_index
        self.stitch_img(self.images[mid_index], self.images[mid_index], final_img, offset, homography)

        homography = np.eye(3)
        for i in range(mid_index, len(self.images) - 1):
            self.i = i + 1
            homography = self.stitch_img(
                self.images[i], self.images[i + 1], final_img, offset, homography
            )

        return None

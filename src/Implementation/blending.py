import cv2
import numpy as np


class Blender:
    def __init__(self, levels=5):
        self.levels = levels

    def gaussianPyramid(self, image):
        pyramid = [image]
        for _ in range(self.levels - 1):
            downsampled = cv2.pyrDown(pyramid[-1])
            pyramid.append(downsampled)
        return pyramid

    def laplacianPyr(self, image):
        pyramid = []
        current_image = image
        for _ in range(self.levels - 1):
            downsampled = cv2.pyrDown(current_image)
            upsampled = cv2.pyrUp(downsampled, dstsize=(current_image.shape[1], current_image.shape[0]))
            laplacian = current_image.astype(float) - upsampled.astype(float)
            pyramid.append(laplacian)
            current_image = downsampled

        pyramid.append(current_image)
        return pyramid

    def getMask(self, image):
        mask = (image[:, :, 0] != 0) & (image[:, :, 1] != 0) & (image[:, :, 2] != 0)
        mask_image = np.zeros(image.shape[:2], dtype=float)
        mask_image[mask] = 1.0
        return mask_image, mask

    def laplacianBlending(self, img1, img2, mask1, mask2):
        laplacian1 = self.laplacianPyr(img1)
        laplacian2 = self.laplacianPyr(img2)

        guassian_mask1 = self.gaussianPyramid(mask1)

        guassian_mask2 = self.gaussianPyramid(mask2)

        blended_pyramid = []
        for lap1, lap2, mask1, mask2 in zip(
            laplacian1, laplacian2, guassian_mask1, guassian_mask2
        ):
            mask1_expanded = np.expand_dims(mask1, axis=-1)
            mask2_expanded = np.expand_dims(mask2, axis=-1)
            blended = lap1 * mask1_expanded + lap2 * mask2_expanded
            blended_pyramid.append(blended)

        return blended_pyramid

    def getBlendedImg(self, blended_pyramid):
        blended_img = blended_pyramid[-1]
        for level in range(len(blended_pyramid) - 2, -1, -1):
            laplacian_level = blended_pyramid[level]
            shape = laplacian_level.shape[:2][::-1]
            blended_img = cv2.pyrUp(blended_img, dstsize=shape).astype(float)
            blended_img += laplacian_level.astype(float)

        return blended_img

    def blend(self, img1, img2):
        _, mask1truth = self.getMask(img1)
        _, mask2truth = self.getMask(img2)

        overlap = mask1truth & mask2truth

        finalMask = mask1truth & ~overlap
        finalMask2 = mask2truth

        # kernel = np.ones((2, 2), np.uint8)
        # expanded_mask1 = (cv2.dilate(finalMask.astype(np.uint8), kernel, iterations=5) > 0 ) & overlap

        blurred_mask = cv2.GaussianBlur(finalMask.astype(np.uint8) * 255, (7, 7), 0)

        _, expanded_mask = cv2.threshold(blurred_mask, 10, 255, cv2.THRESH_BINARY)
        expanded_mask = expanded_mask.astype(bool) & overlap

        finalMask = finalMask | expanded_mask

        mask1 = np.zeros(img1.shape[:2])
        mask2 = np.zeros(img2.shape[:2])
        mask1[finalMask] = 1.0
        mask2[finalMask2 & ~finalMask] = 1.0

        # finalMask = np.zeros_like(tempMask)
        # for y in range(h):
        #   for x in range(w):
        #     if (x - minx) / (maxx - minx + 1e-5) > (y - miny) / (maxy - miny + 1e-5):
        #       finalMask[y, x] = 1.0

        blending_pyramid = self.laplacianBlending(img1, img2, mask1, mask2)
        finalImg = self.getBlendedImg(blending_pyramid)

        return finalImg

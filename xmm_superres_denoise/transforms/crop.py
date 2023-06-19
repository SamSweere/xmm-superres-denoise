import numpy as np


class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_p (float): The crop percentage with respect to the image shape
        mode (str): The cropping mode, options: `random`, `center`
    """

    def __init__(self, crop_p, mode="random"):
        assert isinstance(crop_p, float)
        self.crop_p = crop_p
        self.mode = mode

    def crop_image(self, img, top_p, left_p):
        # top_p: top percentage to crop to
        # left_p: left percentage to crop to
        top = int(top_p * img.shape[0])
        left = int(left_p * img.shape[1])
        h_res = int(img.shape[0] * self.crop_p)
        w_res = int(img.shape[1] * self.crop_p)

        # Check if the top + h_res goes over the image dimensions, if so correct to the left
        if top + h_res > img.shape[0]:
            top -= (top + h_res) - img.shape[0]

        if left + w_res > img.shape[1]:
            left -= (left + w_res) - img.shape[1]

        cropped_img = img[top : top + h_res, left : left + w_res]
        return cropped_img

    def __call__(self, image):
        if self.crop_p == 1.0:
            # Do not crop
            return image

        # Calculate the crop for the lr first since on the hr these values can be uneven
        if self.mode == "random":
            top_p = np.random.uniform(0, 1.0 - self.crop_p)
            left_p = np.random.uniform(0, 1.0 - self.crop_p)
        elif self.mode == "center":
            top_p = (1.0 - self.crop_p) / 2
            left_p = (1.0 - self.crop_p) / 2
        elif self.mode == "boresight":
            # boresight is on (244, 224) on 1x. Having input image of resolution (403, 411) the percentages are:
            top_p = 224.0 / 411.0 - 0.5 * self.crop_p
            left_p = 244.0 / 403.0 - 0.5 * self.crop_p
        else:
            raise ValueError(f"Error, mode {self.mode} unkown")

        if type(image) == list:
            res = []
            for img in image:
                cropped_img = self.crop_image(img, top_p, left_p)
                res.append(cropped_img)
            return res
        else:
            cropped_img = self.crop_image(image, top_p, left_p)
            return cropped_img

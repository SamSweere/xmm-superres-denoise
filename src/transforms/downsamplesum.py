import torch


class DownsampleSum(object):
    """Downsample the image summing the values using conv2d

    Args:
        output_size (int): Desired output size.
    """

    def __init__(self, output_size=None, scale=None):
        # assert isinstance(output_size, int)
        self.output_size = output_size
        self.scale = scale

        if self.output_size and self.scale:
            raise ValueError(
                f"Either specify output size or downsample scale, not both"
            )

        if not self.output_size and not self.scale:
            raise ValueError(f"Either specify output size or downsample scale")

        # Create an input_size and weigths parameter to cache previous weights
        self.input_size = 0
        self.kernel_size = None
        self.weights = None

    def __call__(self, img):
        # Convert the img to tensor
        img = torch.tensor(img)

        input_size = img.shape[-1]

        if self.scale and not self.output_size:
            # Determine the output_size
            self.output_size = input_size * self.scale
            if self.output_size % 2 != 0:
                raise ValueError(
                    f"The intput_size {input_size} with the desired scaling factor {self.scale} does not divide to "
                    f"an integer (output size becomes {self.output_size}"
                )

            self.output_size = int(self.output_size)

        if input_size == self.output_size:
            # The sizes are the same, return itself
            return img

        if (
            input_size != self.input_size
            or self.input_size == 0
            or self.kernel_size is None
            or self.weights is None
        ):
            # New input_size we cannot use the cache
            self.input_size = input_size

            if input_size == self.output_size:
                raise ValueError(
                    f"The desired output size {self.output_size} is the same as the input size {input_size}."
                )

            kernel_size = input_size / self.output_size

            if kernel_size % 2 != 0:
                raise ValueError(
                    f"The desired output size {self.output_size} is not a multiple of 2 of the input size {input_size}"
                )

            # The kernel size seems to be valid, convert it to int
            self.kernel_size = int(kernel_size)
            self.input_size = input_size

            # Generate the weights for the conv2d
            weights = torch.ones((self.kernel_size, self.kernel_size))
            self.weights = weights.view(
                1, 1, self.kernel_size, self.kernel_size
            ).repeat(1, 1, 1, 1)

        # The conv2d needs the data to be in minibatches and have dimensions [1, x, x]
        x = torch.unsqueeze(img, axis=0)
        x = torch.unsqueeze(x, axis=0)

        output = torch.nn.functional.conv2d(x, self.weights, stride=self.kernel_size)

        # Return the result from the minibatch
        return output[0][0].numpy()

import matplotlib.pyplot as plt
import numpy as np
import wandb
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_img(img, title=None, figsize=(10, 10)):
    # Generate and display a test image
    # image = np.arange(65536).reshape((256, 256))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if title is not None:
        ax.set_title(title)
    ax.autoscale(enable=True)

    pos = ax.imshow(img, cmap="hot", interpolation=None)
    fig.colorbar(pos, ax=ax)

    plt.show()


def plot_multiple_img(
    images,
    titles=None,
    rows=None,
    cols=None,
    figsize=(10, 10),
    merge=False,
    colorbar=True,
    save_fig=None,
    show_plot=True,
    wandb_log=False,
):
    """Plot multiple images

    Args:
        images (list): List of images as numpy arrays.
        titles (list) (optional): The titles of the subplots.
        rows (int) (optional): The amount of rows in the subplot, if this is set and cols is None it will expand in the cols direction
        cols (int) (optional): The amount of cols in the subplot, if this is set and rows is None it will expand in the rows direction
        figsize (tuple) (optional): The figsize of the subplots
        merge (boolean) (optional): If this parameter is set all the images are compressed into one big image and displayed
        colorbar (boolean) (optional): If this parameter is set the colorbar is shown
        save_fig (string) (optional): If this parameter is set save the image to a file with the given name
        show_plot (boolean) (optional): If this parameter is set the plot is displayed
        wandb_log (boolean) (optional): If this parameter is set the plot is logged to wandb
    """

    # Determine the rows and cols if they are set to none
    if not rows and not cols:
        # Go for a square layout if possible, if not possible expand in the x direction
        s_dim = np.sqrt(len(images))
        if s_dim.is_integer():
            # The number of images is perfect for a square layout
            rows = int(s_dim)
            cols = int(s_dim)
        else:
            # The number of images is not perfect for a square layout
            # Expand in the x direction
            rows = int(np.floor(s_dim))
            cols = int(np.ceil(len(images) / float(rows)))
    elif not rows and cols:
        # rows is None and cols is not, expand in the rows direction
        rows = int(np.ceil(len(images) / float(cols)))
    elif rows and not cols:
        # cols is None and rows is not, expand in the cols direction
        cols = int(np.ceil(len(images) / float(rows)))
    else:
        if len(images) > rows * cols:
            raise ValueError(
                f"The specified rows ({rows}) and cols ({cols}) do not provide enough room for all the images ({rows * cols}/{len(images)})"
            )

    if merge:
        counter = 0
        img_merged = None
        for y in range(rows):
            # First merge the images in x direction
            img_m_x = None
            for x in range(cols):
                img = images[counter]
                counter += 1

                if img_m_x is None:
                    img_m_x = img
                else:
                    img_m_x = np.hstack((img_m_x, img))

            # Then stack them vertically on the big image
            if img_merged is None:
                img_merged = img_m_x
            else:
                img_merged = np.vstack((img_merged, img_m_x))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.autoscale(enable=True)
        if titles is not None:
            ax.set_title(" - ".join(titles))

        pos = ax.imshow(img_merged, cmap="hot", interpolation=None)
        if colorbar:
            fig.colorbar(pos, ax=ax)

    else:
        fig, axs = plt.subplots(rows, cols, figsize=figsize)

        axs = np.array(axs)
        if len(axs.shape) == 1:
            # We only have one row, expand the dimensions in order to have the same structure
            # as with multiple rows
            axs = np.expand_dims(axs, axis=1)

        counter = 0
        for x in range(axs.shape[0]):
            for y in range(axs.shape[1]):
                #             if rows == 1:
                #                 # This is a fix for when we only have one row
                #                 ax = axs[x][y]
                #             else:
                #                 ax = axs[y][x]
                ax = axs[x, y]

                idx = counter
                counter += 1

                # Check if there are enough images to show, if not continue
                if idx >= len(images):
                    continue

                img = images[idx]
                if titles is not None:
                    ax.set_title(titles[idx])

                pos = ax.imshow(img, cmap="hot", interpolation=None)

                if colorbar:
                    # create an axes on the right side of ax. The width of cax will be 5%
                    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)

                    plt.colorbar(pos, cax=cax)

        # Correct for a nicer layout
        fig.tight_layout()
    #     fig.subplots_adjust(hspace=-0.7)

    if save_fig is not None:
        plt.savefig(save_fig, bbox_inches="tight")
    #         print("Saved figure to",save_fig)

    if wandb_log:
        wandb.log({"images": plt})

    if show_plot:
        plt.show()
    else:
        plt.close()


class Visualizer:
    """This class includes several functions that can display images and print logging information."""

    def __init__(self, configuration):
        """Initialize the Visualizer class.
        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.name = configuration["name"]


# TODO: better implementation
'''
import numpy as np
import sys
from subprocess import Popen, PIPE
import utils
import visdom


class Visualizer():
    """This class includes several functions that can display images and print logging information.
    """

    def __init__(self, configuration):
        """Initialize the Visualizer class.
        Input params:
            configuration -- stores all the configurations
        """
        self.configuration = configuration  # cache the option
        self.display_id = 0
        self.name = configuration['name']

        self.ncols = 0
        self.vis = visdom.Visdom()
        if not self.vis.check_connection():
            self.create_visdom_connections()


    def reset(self):
        """Reset the visualization.
        """
        pass


    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at the default port.
        """
        cmd = sys.executable + ' -m visdom.server'
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Display the current losses on visdom display: dictionary of error labels and values.
        Input params:
            epoch: Current epoch.
            counter_ratio: Progress (percentage) in the current epoch, between 0 to 1.
            losses: Training losses stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'loss_plot_data'):
            self.loss_plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.loss_plot_data['X'].append(epoch + counter_ratio)
        self.loss_plot_data['Y'].append([losses[k] for k in self.loss_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.loss_plot_data['X'])] * len(self.loss_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.loss_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.loss_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_current_validation_metrics(self, epoch, metrics):
        """Display the current validation metrics on visdom display: dictionary of error labels and values.
        Input params:
            epoch: Current epoch.
            losses: Validation metrics stored in the format of (name, float) pairs.
        """
        if not hasattr(self, 'val_plot_data'):
            self.val_plot_data = {'X': [], 'Y': [], 'legend': list(metrics.keys())}
        self.val_plot_data['X'].append(epoch)
        self.val_plot_data['Y'].append([metrics[k] for k in self.val_plot_data['legend']])
        x = np.squeeze(np.stack([np.array(self.val_plot_data['X'])] * len(self.val_plot_data['legend']), 1), axis=1)
        y = np.squeeze(np.array(self.val_plot_data['Y']), axis=1)
        try:
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' over time',
                    'legend': self.val_plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'metric'},
                win=self.display_id+1)
        except ConnectionError:
            self.create_visdom_connections()


    def plot_roc_curve(self, fpr, tpr, thresholds):
        """Display the ROC curve.
        Input params:
            fpr: False positive rate (1 - specificity).
            tpr: True positive rate (sensitivity).
            thresholds: Thresholds for the curve.
        """
        try:
            self.vis.line(
                X=fpr,
                Y=tpr,
                opts={
                    'title': 'ROC Curve',
                    'xlabel': '1 - specificity',
                    'ylabel': 'sensitivity',
                    'fillarea': True},
                win=self.display_id+2)
        except ConnectionError:
            self.create_visdom_connections()


    def show_validation_images(self, images):
        """Display validation images. The images have to be in the form of a tensor with
        [(image, label, prediction), (image, label, prediction), ...] in the 0-th dimension.
        """
        # zip the images together so that always the image is followed by label is followed by prediction
        images = images.permute(1,0,2,3)
        images = images.reshape((images.shape[0]*images.shape[1],images.shape[2],images.shape[3]))

        # add a channel dimension to the tensor since the excepted format by visdom is (B,C,H,W)
        images = images[:,None,:,:]

        try:
            self.vis.images(images, win=self.display_id+3, nrow=3)
        except ConnectionError:
            self.create_visdom_connections()


    def print_current_losses(self, epoch, max_epochs, iter, max_iters, losses):
        """Print current losses on console.
        Input params:
            epoch: Current epoch.
            max_epochs: Maximum number of epochs.
            iter: Iteration in epoch.
            max_iters: Number of iterations in epoch.
            losses: Training losses stored in the format of (name, float) pairs
        """
        message = '[epoch: {}/{}, iter: {}/{}] '.format(epoch, max_epochs, iter, max_iters)
        for k, v in losses.items():
            message += '{0}: {1:.6f} '.format(k, v)

        print(message)  # print the message
'''

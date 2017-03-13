import numpy as np
from cv2 import filter2D

class Descriptor:

    def __init__(self, imgs):
        if len(imgs.shape) == 4:
            self.samples = np.array([self._getHOGs(img) for img in imgs])
        else:
            self.samples = self._getHOGs(imgs)

    def _getHOGs(self, img):
        # 1. Compute gradients and their orientations.
        gradfilter = np.array([[-1, 0, 1]], dtype='float32')
        xgrads = filter2D(img.copy(), -1, gradfilter)
        ygrads = filter2D(img.copy(), -1, gradfilter.T)
        gradmag = np.sqrt(xgrads**2 + ygrads**2)
        maxmask = self._max_channel_mask(gradmag)
        grador = np.arctan(np.divide(xgrads, ygrads+.1e-10))
        grador = (grador*maxmask).sum(axis=2)
        gradmag = (gradmag*maxmask).sum(axis=2)
        # 2. Magnitude-weighted voting of orientations into histogram bins.
        hist_img = self._compute_histograms(grador, gradmag)
        # 3. Aggregate bin votes over 8x8 cells.(This desperately needs to be clearer!)
        cell_hist = np.zeros([16, 8, 9])
        for row_ix in np.arange(0, 128, 8):
            for col_ix in np.arange(0, 64, 8):
                cell_pixels = hist_img[row_ix:row_ix+8, col_ix:col_ix+8, :]
                cell_hist[int(row_ix/8), int(col_ix/8), :] = np.tensordot(cell_pixels, np.ones([8, 8, 1]),
                                                                                axes=((0, 1), (0, 1))).squeeze()
        # 4. Normalize each cell's histogram against its neighbors', then combine the results into a descriptor vector.
        descriptor = np.zeros([14*6, 81])
        for row in range(14):
            for col in range(6):
                block = cell_hist[row:row+3, col:col+3, :].flatten()
                block_descriptor = block/np.linalg.norm(block) #Normalization
                descriptor[row*6 + col, :] = block_descriptor
        # 5. Return the result!
        return descriptor.flatten()

    def _max_channel_mask(self, img): # For getting the maximum pixels' mask from a 3D image
        z = np.zeros(img.shape[:2])
        comparator = np.expand_dims(np.argmax(img, axis=2), axis=2)
        mask = (np.stack([z, z+1, z+2], axis=2) == comparator)
        return mask

    def _compute_histograms(self, grador, gradmag): # For getting the orientation votes at each pixel in an image
        bins = np.linspace(-np.pi/2, np.pi/2, 9)
        bin_width = bins[1] - bins[0]
        hist_img = np.tile(bins, [grador.shape[0], grador.shape[1], 1])
        mags = np.expand_dims(gradmag, axis=2)
        ors = np.expand_dims(grador, axis=2)
        mask = abs(ors - hist_img) < bin_width
        hist_img = mask*mags*(1 - abs(ors - hist_img)/bin_width)
        # ^ Distribute votes proportionally between directly adj. bins
        return hist_img

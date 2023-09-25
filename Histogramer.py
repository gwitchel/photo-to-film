import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
class Histogramer:
    skin_tones = None
    source_px = []
    target_px = None
    target_link = None
    def __init__(self,src_images,target) -> None:
        self.target_link = target
        self.target_px = cv2.imread(target)[:,:,::-1]
        dims = self.target_px.shape
        self.find_faces()
        print(dims)
        for image in src_images: 
            img = cv2.imread(image)[:,:,::-1]
            resized_img = cv2.resize(img, (dims[1],dims[0]), interpolation = cv2.INTER_AREA)
            self.source_px.append(resized_img) 

    def show_conversion(self,reference,matched):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, 
                                    figsize=(8, 3),
                                    sharex=True, sharey=True)
    
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()
        
        ax1.imshow(self.target_px)
        ax1.set_title('Source')
        ax2.imshow(reference)
        ax2.set_title('Reference')
        ax3.imshow(matched.astype(int))
        ax3.set_title('Matched')

        plt.tight_layout()
        plt.show()

    def convert_image(self):
        ## Converts an image using the histogram of best fit 
        lowest_err = math.inf
        best_match = None
        best_source = None
        for img in self.source_px: 
            matched = self.match_histograms(self.target_px,img)
            err = np.sum(np.linalg.norm(matched-self.source_px))
            if err < lowest_err: 
                lowest_err = err
                best_match = matched 
                best_source = img
        
        return best_source,best_match,lowest_err
        self.show_conversion(best_source,best_match)
        

    def _match_cumulative_cdf(self,source, template):
        """
        Return modified source array so that the cumulative density function of
        its values matches the cumulative density function of the template.
        """
        if source.dtype.kind == 'u':
            src_lookup = source.reshape(-1)
            src_counts = np.bincount(src_lookup)
            tmpl_counts = np.bincount(template.reshape(-1))

            # omit values where the count was 0
            tmpl_values = np.nonzero(tmpl_counts)[0]
            tmpl_counts = tmpl_counts[tmpl_values]
        else:
            src_values, src_lookup, src_counts = np.unique(source.reshape(-1),
                                                        return_inverse=True,
                                                        return_counts=True)
            tmpl_values, tmpl_counts = np.unique(template.reshape(-1),
                                                return_counts=True)

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        return interp_a_values[src_lookup].reshape(source.shape)
    
    def match_histograms(self,image, reference, channel_axis=None):
        """Adjust an image so that its cumulative histogram matches that of another.

        The adjustment is applied separately for each channel.

        Parameters
        ----------
        image : ndarray
            Input image. Can be gray-scale or in color.
        reference : ndarray
            Image to match histogram of. Must have the same number of channels as
            image.
        channel_axis : int or None, optional
            If None, the image is assumed to be a grayscale (single channel) image.
            Otherwise, this parameter indicates which axis of the array corresponds
            to channels.

        Returns
        -------
        matched : ndarray
            Transformed input image.

        Raises
        ------
        ValueError
            Thrown when the number of channels in the input image and the reference
            differ.

        References
        ----------
        .. [1] http://paulbourke.net/miscellaneous/equalisation/

        """
        if image.ndim != reference.ndim:
            raise ValueError('Image and reference must have the same number '
                            'of channels.')

        if channel_axis is not None:
            if image.shape[-1] != reference.shape[-1]:
                raise ValueError('Number of channels in the input image and '
                                'reference image must match!')

            matched = np.empty(image.shape, dtype=image.dtype)
            for channel in range(image.shape[-1]):
                matched_channel = self._match_cumulative_cdf(image[..., channel],
                                                        reference[..., channel])
                matched[..., channel] = matched_channel
        else:
            # _match_cumulative_cdf will always return float64 due to np.interp
            matched = self._match_cumulative_cdf(image, reference)

        return matched

    def find_faces(self):
        img = cv2.imread(self.target_link)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        face = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.5, minNeighbors=5, minSize=(40, 40)
        )

        #draw a bounding box 
        for (x, y, w, h) in face:
            # TODO: create seperate histograms for different faces
            print("FOUND A FACE")
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        plt.figure(figsize=(20,10))
        plt.imshow(img_rgb)
        plt.axis('off')

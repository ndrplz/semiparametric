from PIL import Image

import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F


class MyRandomAffine(transforms.RandomAffine):

    def sample_params(self, w, h):
        self.size = (h, w)
        self.ret = self.get_params(self.degrees, self.translate, self.scale,
                                   self.shear, (w, h))

    def __call__(self, *in_images, fillcolor=None):
        # Check that all input images share the same size which also
        #  match the declared one
        assert np.all([i.shape[:2] == self.size for i in in_images])

        fill = fillcolor
        if fillcolor is None:
            fill = self.fillcolor

        output = []
        for image in in_images:
            image = Image.fromarray(image)
            x = F.affine(image, *self.ret, resample=self.resample,
                         fillcolor=fill)
            output.append(np.asarray(x))

        if len(output) == 1:
            output = output[0]
        else:
            output = np.stack(output, axis=0)

        return output

# Semi-parametric Object Synthesis (Paper ID: 265)

Here you can find the code and supplementary material of ICCV2019 submission: **Semi-parametric Object Synthesis**.

<p align="center">
  <img src="gifs/teaser_car.gif"/ alt="Rotating cars">
</p>

## Code

### How to run

To run our demo code, you need to download the following.
- Pascal3D+ vehicles dataset ([here](https://drive.google.com/open?id=1tP0MNK-505d8OWoyIp267JhkIfJt7jh1))
- 3D CADs ([here](https://drive.google.com/open?id=1V5sysWzg-jVfY50cYZzjg6cgoIgzu8u0))
- Pre-trained weights ([here](https://drive.google.com/open?id=1rF5sz_kXMmcu7wK9e6_PLqTh59GGIbuq))

### Description and usage

Here you can find the instructions to run our demo code. The entry point is [`run_rotate.py`](https://github.com/iccv19sub265/semiparametric/blob/master/run_rotate.py).
When you run it, you should see a GUI like the following:

<p align="center">
  <img src="imgs/viewport.png"/ alt="Viewport" width="80%">
</p>

The GUI is composed of two windows: the *viewport* and the *output* one.

While the focus is on the *viewport*, keyboard can be used to move around the object in spherical coordinates. [Here](https://github.com/iccv19sub265/semiparametric/blob/master/help.txt) the full list of commands is provided. While you move, the *output* shows both Image Completion Network (ICN) inputs (2.5D sketches, appearance prior) and network prediction. Please refer to Sec.3 of the paper for details.

*Notice*: it may happen that when starting the program, open3D does not render anything. This is an initialization issue. In case this happens, just focus on the *viewport* and press spacebar a couple of times until you see both windows rendered properly.

## Supplementary Material

### Extreme viewpoint transformations (see Sec. 4)

Due to its *semi-parametric* nature, our method is much more robust than competitors to extreme viewpoint changes.

Here they are some examples:

<p align="center">
  <img src="gifs/gif_zoom.gif"/ alt="Zoom gif" width="40%">
  </br> Manipulation of radial distance.
</p>

<p align="center">
  <img src="gifs/gif_elevation.gif"/ alt="Elevation gif" width="40%">
  </br> Manipulation of elevation.
</p>

<p align="center">
  <img src="gifs/gif_pickup.gif"/ alt="Rototranslation gif" width="40%">
  </br> Arbitrary rototranslation.
</p>

### Data augmentation (see Sec. 4.4)

Additional examples generated synthetically using our model are shown below.

Each row is generated as follows. Given an image from [Pascal3D+](http://cvgl.stanford.edu/projects/pascal3d.html), other examples in the same pose are randomly sampled from the dataset. Then, our method is used to transfer the appearance of the latter to the pose of the first. Eventually, generated vehicles are stiched upon the original image. For a seamless collaging, we perform a small Gaussian blur at the mask border.

<p align="center">
  <img src="imgs/aug_data_supp.jpg"/ alt="Generated data" width="80%">
</p>

Percentage of Correct Keypoints (PCK) logged in TensorBoard during training (see Sec. 4.4)
<p align="center">
  <img src="imgs/pck_graph.png"/ alt="PCK graph" width="80%">
</p>



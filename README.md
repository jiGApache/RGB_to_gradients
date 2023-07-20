# RGB to gradients
Calculating the brightness gradient of an image along the horizontal and vertical axes

The entire model consists of 2 convulutional layers:
1. Performs converting RGB image to grayscale image according to [formula of relative luminance](https://en.wikipedia.org/wiki/Relative_luminance)
2. Performs gradient calculation using 3x3 [Sobel operators](https://en.wikipedia.org/wiki/Sobel_operator).

### Usage examples
The gradient calculation performs from images placed in ```Images``` folder. 

To start gradient calculation run ```python main.py```.

![Alt text](https://github.com/jiGApache/RGB_to_gradients/raw/main/Results/Figure_1.png)

Optional flag ```-e``` or ```--detect-edges``` allows to detect edges from computed gredients.

![Alt text](https://github.com/jiGApache/RGB_to_gradients/raw/main/Results/Figure_2.png)
# REMOTE SENSING IMAGE ACQUISITION, ANALYSIS AND APPLICATIONS - Week 3

Course tutor: John Richards

All original diagrams © Henry Pan. Course slides are used under fair use for educational purposes. Not for commercial use.

## Lecture 9. Correcting geometric distortion using mapping functions and control points

Two requirements

- Have available a map of the region which is correct geometrically
- Identify points—called control points—in both the map and the image

![How the mapping functions are used](RS%20week%203.assets/How%20the%20mapping%20functions%20are%20used.png)

### How to determine the unknown coefficients in the mapping polynomials

| Polynomial Order $n$     | Terms (basis functions)                                      | Number of Coefficients per equation | Minimum GCPs required |
| ------------------------ | ------------------------------------------------------------ | ----------------------------------- | --------------------- |
| 1 (First-order / Linear) | 1, u, v                                                      | 3                                   | 3                     |
| 2 (Second-order)         | 1, u, v, u^2, uv, v^2                                        | 6                                   | 6                     |
| 3 (Third-order)          | 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3                  | 10                                  | 10                    |
| 4 (Fourth-order)         | 1, u, v, u^2, uv, v^2, u^3, u^2v, uv^2, v^3, u^4, u^3v, u^2v^2, uv^3, v^4 | 15                                  | 15                    |

$$
N_{\text{coeff}} = \frac{(n+1)(n+2)}{2}
$$

> Usually, we choose more control points than necessary and generate least squares estimates of the unknown coefficients

### Compare the benefits and limitations with using very high degree mapping polynomials as against first order mapping polynomials

![output (9)](RS%20week%203.assets/output%20(9).png)

- High-order polynomials fit control points very accurately but behave unpredictably outside their range, leading to poor extrapolation
- Low-order polynomials fit less precisely near control points but provide smoother, more stable extrapolation

## Lecture 10. Resampling

> The process of finding an actual value for the pixel in the image to place on to the map grid using the process of mapping polynomials is called resampling

![The main three methods of resampling](RS%20week%203.assets/The%20main%20three%20methods%20of%20resampling.png)

Convolution can be thought of as scanning an image with a lens (the kernel), whose curvature determines how each location is re-weighted — with the center receiving more influence than the edges — and whose shape can vary, much like choosing between different types of lenses

![How to understand the Convolution](RS%20week%203.assets/How%20to%20understand%20the%20Convolution.png)

## Lecture 11. An image registration example

> One image (called the master) can take the place of the map, and the other image(called the slave) can be registered to i

Control points should be well distributed, and enclose the region of the image for which accurate registration is important

> As much as possible, those sorts of control points should be on headlands and not river inlets so that tidal variations are minimized

![image-20250812184308248](RS%20week%203.assets/image-20250812184308248.png)

*image source: slide 1.11.7 of this course*

When the distribution of control points is not good, using low-degree fitting yields better results

**Question: does relative scale matter?**

For a higher order resampling technique, it is needed

**Question: Would you choose one image as a master and then register the other four to it,or would you register image 2 to image 1 and then image 3 to the newly registered image 2, and so on?**

![Chain registration will amplify errors](RS%20week%203.assets/Chain%20registration%20will%20amplify%20errors.png)

## Lecture 12. How can images be interpreted and used?

there are two approaches to interpretation:

- photointerpretation: rely on experts
- Quantitative analysis/ machine learning

![How to form a color image in Remote Sensing](RS%20week%203.assets/How%20to%20form%20a%20color%20image%20in%20Remote%20Sensing.png) 

## Lecture 13. Enhancing image contrast

**Question: Why are the original satellite remote sensing images all reddish?**

![Why are the original satellite remote sensing images all reddish](RS%20week%203.assets/Why are the original satellite remote sensing images all reddish-1756459701480-2.png)

As shown in the figure, the red channel accounts for approximately 55%, thus exerting an overwhelming impact on the coloration of the image

Therefore, most images need to enhance contrast through **the brightness value mapping function**, making the final image products easier to understand

![image-20250813223437178](RS%20week%203.assets/image-20250813223437178.png)

*image source: slide 1.13.7 of this course*

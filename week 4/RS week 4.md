# REMOTE SENSING IMAGE ACQUISITION, ANALYSIS AND APPLICATIONS - Week 4

Course tutor: John Richards

All original diagrams © Henry Pan. Course slides are used under fair use for educational purposes. Not for commercial use.

## Lecture 14. An introduction to classification (quantitative analysis)

> The process of using pixels for which the class labels are known to find regions in the spectral space corresponding to each class is called “training”

First step: training the classifier

![The foundation of machine learning in RS](RS%20week%204.assets/The%20foundation%20of%20machine%20learning%20in%20RS.png)

- Each pixel is described by a set of brightness values from multiple spectral bands (**pixel vector**)
- Known pixels from ground truth are used to train the classifier and set decision boundaries in **spectral space**
- Unknown pixels are mapped into the same space and assigned to a class based on their position
- The output is a **thematic map** showing class labels (e.g., vegetation, soil, water) for all pixels.

**Question: In two dimensions, how do you think we might find the lines that separate the three classes in the example in this lecture**

![A simple approach is to find the mean vector of each class](RS%20week%204.assets/A%20simple%20approach%20is%20to%20find%20the%20mean%20vector%20of%20each%20class.png)

## Lecture 15. Classification: some more detail

| Photointerpretation(human analyst)            | Quantitative analysis (computer)              |
| --------------------------------------------- | --------------------------------------------- |
| shape determination is easy                   | shape determination is complex                |
| spatial information is easy to use in general | spatial decision making in general is limited |

The conclusion that "computers are good at pixels but not at shape and spatial decision-making" has been significantly weakened by deep learning (UNet/Transformer/instance segmentation), self-supervised/multimodal learning, and foundation models

However, due to the physical upper limit of resolution, cross-sensor generalization, and uncertainty quantification, "human-machine collaboration" is still necessary

### Supervised Learning

- Analyst acquires labelled training pixels 
- The labelled pixels are used to train the classifier (segment the spectral space)
- Unknown pixels are presented to the classifier, one by one, and labelled according to the region of spectral space in which they fall
- The full image is now labelled, so that a thematic map has been produced. Pixels with the same labels can be counted to give area estimates

```
"Obtain annotations → Train classifiers → Infer on the entire graph"
```

But now Unsupervised Learning is becoming more powerful, e.g., DINOv3

```
"Large-scale self-supervised pre-training → Rapid fine-tuning (or zero-shot) for downstream tasks → Infer on the entire graph"
```

In remote sensing, the high dimensionality of spectral data makes linear algebra essential for describing and manipulating data spaces beyond human visual intuition, enabling machine learning algorithms to operate in hyperspectral domains

## Lecture 16. Correlation and covariance

![output (11)](RS%20week%204.assets/output%20(11).png)

> The mean vector tells us the average position of pixels in the spectral domain. We would also like to know how they spread about that **mean position**. That is the role of the **covariance matrix**, which is the multidimensional form of the variance (square of standard deviation) of the normal distribution in one dimension

$$
\mathbf{C}_x = \frac{1}{K - 1} \sum_{k=1}^{K} (\mathbf{x}_k - \mathbf{m})(\mathbf{x}_k - \mathbf{m})^T
$$

When estimating variance or covariance from sample data, dividing by K−1 rather than 𝐾 yields an unbiased estimator by correcting the downward bias introduced when the sample mean is used in place of the true population mean

![output (12)](RS%20week%204.assets/output%20(12).png)
$$
r_{ij} = c_{ij} / \sqrt{c_{ii} c_{jj}}
$$
This step is equivalent to eliminating the influence of dimensions using the variable's own scale (standard deviation - 标准差), turning the result into a unitless pure correlation coefficient within the range of [-1, 1]

## Lecture 17. The principal components transform

Common image transformations encountered in remote sensing are:

- Principle components transform / principle components analysis
- Fourier transform 
- Wavelet transform
- Band arithmetic 

### The formula derivation of PCA

$$
\mathbf{C}_y = \mathcal{E} \left\{ (\mathbf{y} - \mathbf{m}_y)(\mathbf{y} - \mathbf{m}_y)^T \right\}①
$$

We know:
$$
\mathbf{y} = \mathbf{G} \mathbf{x}
$$
Then:
$$
\mathbf{m}_y = \mathcal{E} \{\mathbf{y}\} = \mathcal{E} \{\mathbf{G}\mathbf{x}\} = \mathbf{G}\mathcal{E} \{\mathbf{x}\} = \mathbf{G}\mathbf{m}_x ②
$$
Because of ① and ②
$$
\mathbf{C}_y = \mathcal{E} \left\{ (\mathbf{G}\mathbf{x} - \mathbf{G}\mathbf{m}_x)(\mathbf{G}\mathbf{x} - \mathbf{G}\mathbf{m}_x)^T \right\} = \mathbf{G}\mathcal{E} \left\{ (\mathbf{x} - \mathbf{m}_x)(\mathbf{x} - \mathbf{m}_x)^T \right\} \mathbf{G}^T = \mathbf{G}\mathbf{C}_x \mathbf{G}^T
$$

### The geometric intuition of PCA

![Geometric intuition of principal components analysis (PCA)](RS%20week%204.assets/Geometric%20intuition%20of%20principal%20components%20analysis%20(PCA)%20.png)

- PCA is a rotation of the coordinate system that transforms correlation into uncorrelation
- The shape of the ellipse remains unchanged; it only gets "aligned"
- The direction of the first principal component (corresponding to the largest eigenvalue) is equal to the direction of the major axis of the ellipse. Generally, in the two-dimensional case, the two principal axes of the ellipse are exactly the two eigenvectors of the covariance matrix

In summary, **Principal Component Analysis (PCA) can be understood from three perspectives**: 

- geometrically, it rotates the coordinate axes to align with the principal directions of the data
- algebraically, it diagonalizes the covariance matrix, eliminating cross terms
- in terms of projection, it constructs a new orthogonal basis where the projected variables are uncorrelated.

**Question:  is the order of the product important?**

matrix multiplication and vector multiplication are non-commutative, the order is very important

![Dot Product vs Outer Product](RS%20week%204.assets/Dot%20Product%20vs%20Outer%20Product.png)

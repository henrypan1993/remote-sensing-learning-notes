# REMOTE SENSING IMAGE ACQUISITION, ANALYSIS AND APPLICATIONS - Week 9

Course tutor: John Richards

All original diagrams © Henry Pan. Course slides are used under fair use for educational purposes. Not for commercial use.

### Lecture 14. Deep learning and the convolutional neural network, part 1

> we are going to commence by considering a problem in remote sensing image classification that has been of interest since the1970s-i.e., being sensitive to surrounding pixels when choosing the best class label for a pixel of interest. This is known as (spatial) context classification

with the algorithm developed, scientist began to consider the neighbors of a pixel: point classifier → context classifier

| Method                                                       | Definition                                                   | Approx. Time Proposed                                        | Limitations                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ECHO (Extraction and Classification of Homogeneous Objects)** | Groups pixels into regions based on spectral similarity before classification, then labels the regions instead of individual pixels. | Early 1980s (emerged with region-based image segmentation in remote sensing) | Sensitive to boundaries; region growing can introduce errors; relatively high computational cost. |
| **Relaxation Labelling**                                     | Represents each pixel’s class as posterior probabilities, which are iteratively updated by referring to the posterior probabilities of neighboring pixels. | 1970s (first in pattern recognition, later applied to remote sensing) | Strongly depends on initial classification; may converge to wrong results; computationally intensive due to iterations. |
| **Computing Measures of Texture**                            | Calculates texture features (e.g., smoothness, roughness, contrast) in the neighborhood of each pixel and uses them as additional features in a classifier. | 1980s–1990s (introduced with advances in image processing)   | Highly sensitive to window size; texture varies across scales, making parameter selection difficult. |

![](RS%20week%209.assets/The%20analogies%20of%20Classical%20context%20classification.png)

In previous treatments, we assumed only one pixel at a time, described by its feature vector, was fed into the network. In convolutional neural networks we allow for the whole image to be fed into the network at once

![From the perspective of problem solving to see the emergence of CNN](RS%20week%209.assets/From%20the%20perspective%20of%20problem%20solving%20to%20see%20the%20emergence%20of%20CNN.png)

## Lecture 15. Deep learning and the convolutional neural network, part 2

> In spatial convolution a window, called a kernel, is moved over an image row by row and column by column, A new brightness value is created for the pixel under the center of the kernel by taking the products of pixel brightness values and the kernel entries, and then summing the result

> In the CNN the kernel is usually called a filter, and the set of input pixels covered by the filter is called a local receptive
> field.

**CNNs and Spatial Structure**

- CNNs reduce the number of parameters by using kernels instead of full connections
- A kernel (spatial neighborhood of weights) determines what feature is extracted
- Example: an edge-detection filter assigns a strong positive weight to the center pixel and negative weights to neighbors, highlighting boundaries in the image

### Some common concepts

![Three key concepts in CNNs stride, pooling, and flattening ](RS%20week%209.assets/Three%20key%20concepts%20in%20CNNs%20stride,%20pooling,%20and%20flattening%20.png)

**Stride**: which is the offset of the filter position (or receptive field) that provides input to each successive node in the hidden laver

**Pooling**: 

Pooling is the process of aggregating values within a local neighborhood of a feature map (e.g., a $2 \times 2$ region) into a single representative value.

Common methods:

- **Max pooling**: selects the maximum value → preserves the strongest feature.
- **Average pooling**: takes the average value → preserves the overall trend.

Word Meaning

- The word *pool* means “to collect or combine things together.”
- Example from the dictionary: *“The students pooled their money to buy a gift for the teacher.”*
   → By analogy, pooling in CNNs collects neighboring pixel values into one.

**Flattening**: Flattening is the operation of converting a multi-dimensional array (e.g., 2D matrix or 3D tensor) into a one-dimensional vector

### Outputs

| **Option**             | **Description**                                              |
| ---------------------- | ------------------------------------------------------------ |
| 1. Deeper Network      | Feed pooled outputs into another convolutional layer to extract more complex features |
| 2. Direct Output       | Pass pooled outputs directly to the output layer             |
| 3. Feature Selector    | Use pooled outputs as inputs to a fully connected NN, where CNN acts as a feature extractor |
| 4. Class Probabilities | Generate a set of class probabilities (e.g., via softmax)    |

$$
p(o_n = t_n) = \frac{e^{o_n}}{\sum_{n=1}^{N} e^{o_n}}
$$

Note:

$$
\sum_{n} p_n = 1
$$

and 

$$
p \in (0,1)
$$

Softmax converts a vector of raw outputs (logits) into a probability distribution

Softmax = exponentiation + normalization

![Argmax vs softmax in machine learning](RS%20week%209.assets/Argmax%20vs%20softmax%20in%20machine%20learning.png)

Key Ideas:

- Exponentiation: ensures all values are positive and amplifies differences between larger and smaller logits.

- Normalization: divides by the sum of all exponentiated values so that the probabilities add up to 1.

> Whereas the sigmoid function had been a popular activation function with NNs in the past, almost always CNNs now use the rectified linear unit (ReLU) activation function; that improves training.

| **Aspect**         | **Sigmoid**                                                  | **ReLU**                                   |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------ |
| **Formula**        | $$\\sigma(z) = \frac{1}{1 + e^{-z}}\$$                         | $$f(z) = \max(0, z)$$                      |
| **Computation**    | Requires exponential (exp), relatively slow                  | Simple comparison, very fast               |
| **Gradient**       | $$\sigma(z)(1-\sigma(z))$$, can vanish when $$\(z \ll 0\) or \(z \gg 0\)$$ | Derivative is 1 for $$\(z>0\)$$, 0 for $$\(z<0\)$$ |
| **Training Speed** | Slower due to vanishing gradient                             | Faster convergence, stable updates         |
| **Issues**         | Vanishing gradients, saturation                              | Dead neurons (if many $$\(z<0\)$$)             |

## Lecture 16. Deep learning and the convolutional neural network, part 3

A color image can be represented by three channels: **Red (R), Green (G), and Blue (B)**.

- Each pixel is described as:
  - $I(r, i, j)$ for the red channel
  - $I(g, i, j)$ for the green channel
  - $I(b, i, j)$ for the blue channel
- The convolution is applied **separately** on each channel with its corresponding kernel weights:

$$
r1 = \sum_{i,j} I(r, i, j)\, w(r, i, j)
$$

$$
g1 = \sum_{i,j} I(g, i, j)\, w(g, i, j)
$$

$$
b1 = \sum_{i,j} I(b, i, j)\, w(b, i, j)
$$

- The three results are then **summed together**, with a bias $\theta$ added, and passed through an **activation function** (e.g., ReLU).
- This process is repeated simultaneously for all kernel positions as the filter slides over the image.
- Important: With 3 channels, the kernel has **three times more parameters** compared to a single-channel (grayscale) image.

A convolutional layer uses multiple filters of different sizes in parallel so that each filter captures distinct features (e.g., shapes or details), providing richer information for deeper layers

![Typical CNNs topology](RS%20week%209.assets/Typical%20CNNs%20topology.png)

- **Pipeline**
  1. **Input image** ($N \times N$ pixels)
  2. **Convolution + Activation** (extract features with multiple filters)
  3. **Pooling** (reduce spatial resolution, retain important info)
  4. **Convolution + Activation** (deeper feature extraction)
  5. **Pooling**
  6. **Flattening** (convert feature maps into a vector)
  7. **Fully Connected Neural Network** (classification or regression)
- **Notes**
  - This block-diagram style is the **standard way** to represent CNNs in textbooks and papers.
  - It captures the **core idea of LeNet/AlexNet** and remains the backbone of many modern CNN architectures.
  - Sometimes there are **cross-connections** (e.g., ResNet, DenseNet), but the basic flow stays the same.

📌 **Key takeaway**: This simplified representation is a **classic modeling approach** for CNNs, widely used as the foundation for both teaching and research

There are two new concepts:

**Overfitting**

- Definition: Model performs well on the training set but poorly on the test set.
- Interpretation: High fitting, poor generalization.
- Cause: Model is too complex or trained too long, memorizing noise/details instead of general patterns.

**Underfitting**

- Definition: Model performs poorly on both the training set and the test set.
- Interpretation: Poor fitting, poor generalization.
- Cause: Model is too simple or insufficiently trained, failing to capture the underlying structure.

### Analyze hyperspectral data for spatial properties alone

![Two CNN Approaches for Hyperspectral Data Analysis](RS%20week%209.assets/Two%20CNN%20Approaches%20for%20Hyperspectral%20Data%20Analysis.png)

**Q1. Why apply PCA before CNN in hyperspectral images?**

- Hyperspectral images often contain hundreds of spectral bands (e.g., 200+).

- Directly applying convolution kernels to all bands leads to a parameter explosion:

$$\text{params} \propto (\text{bands}) \times (\text{kernel size}) \times (\text{filters})$$

- PCA can reduce dimensionality by keeping only the most significant components, greatly reducing the number of input channels.

- This lowers computation cost and mitigates overfitting, while still preserving most of the information.

**Q2. How many principal components (PCs) should be kept?**

- PCA eigenvalues typically show a **sharp drop followed by a long tail**.
- Keeping enough PCs to preserve **~95% of variance** is a common rule of thumb.
- The required number $n$ is **data-dependent**:
  - In many cases, **3–5 PCs** may already capture most variance.
  - For tasks requiring finer spectral detail, **10–20 PCs** may be needed.
- Important: The curve is not strictly like $y = 1/x$; the decay pattern depends on the dataset.

### Analyze hyperspectral data for spectral properties alone

- **Spatial Convolution (traditional CNN):**
  - Kernel slides over **i, j** (spatial dimensions).
  - $n$ (spectral bands) are treated as channels combined in the convolution.
- **Spectral Convolution:**
  - Ignore **i, j** (no spatial context).
  - Focus on each pixel’s **$n$-dimensional spectral vector**.
  - Kernel slides along **$n$** (spectral dimension) to capture band correlations.

👉 Essentially, this is **1D convolution in the spectral domain** vs. **2D convolution in the spatial domain**.

---

**Q1. If you wished to reduce the dimensionality of hyperspectral data by using a principal components transform prior to inputting it into a CNN, what measure might you use to determine how many principal components to retain?**

**A:**  Use the **explained variance ratio**:

- Compute eigenvalues of the covariance matrix.  
- Select the smallest number of PCs such that the cumulative variance retained meets a chosen threshold (e.g., 95%).  

$$
\text{Explained Variance Ratio}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^n \lambda_i}
$$

**Q2. The diagram below shows a three-dimensional convolution kernel. Describe its operation mathematically. Is it any different from applying three separate two-dimensional kernels and summing the results?**

![image-20250901180417636](RS%20week%209.assets/image-20250901180417636.png)

**A:**  Mathematical operation:  
$$R(x,y) = \sum_{c=1}^{C} \sum_{i,j} w(c,i,j) \cdot I(c, x+i, y+j)$$

- Equivalent to applying a 2D convolution on each channel separately and summing the results.  
- No difference: this is exactly how standard multi-channel convolution works in CNNs (plus bias and activation).  

---

## Lecture 17. CNN examples in remote sensing

Hu, W., Huang, Y., Wei, L., Zhang, F., & Li, H. (2015). Deep Convolutional Neural Networks for Hyperspectral Image Classification. *Journal of Sensors*, *2015*, 1–12. https://doi.org/10.1155/2015/258619

![image-20250901223238080](RS%20week%209.assets/image-20250901223238080.png)

| Layer                  | Description                           | Formula                         | Value |
| ---------------------- | ------------------------------------- | ------------------------------- | ----- |
| **Input**              | Spectral bands per pixel              | $n_1$                           | $220$ |
| **Convolution C1**     | Kernel size $24 \times 1$, stride = 1 | $n_2 = n_1 - 24 + 1$            | $197$ |
| **Pooling M2**         | Max pool kernel $5 \times 1$          | $n_3 = \lfloor n_2 / 5 \rfloor$ | $40$  |
| **Flatten**            | Feature maps × nodes                  | $20 \times n_3$                 | $800$ |
| **Fully Connected F3** | Hidden layer nodes (hyperparameter)   | $n_4$                           | $100$ |
| **Output F4**          | Number of classes                     | $n_5$                           | $8$   |

To verify 81,408 unknowns:

- **Conv layer**: 500 (24 weights + 1 offset per filter).
- **Conv-pooling**: 0 new unknowns.
- **Fully-connected (hidden) layer**: \((20×40 + 1)×100 = 80,100\).
- **Hidden-output layer**: \((100 + 1)×8 = 808\).
- **Total**: \(500 + 80,100 + 808 = 81,408\).

![image-20250901223845570](RS%20week%209.assets/image-20250901223845570.png)

From left to right: ground truth, RBF-SVM, and CNNs

Accuracy comparison

- CNN: 90.2%
- SVM: 87.6%

> Had it incorporated spatial filtering too, we would expect to see a much cleaner the thematic map

---

Yang, J., Zhao, Y.-Q., & Chan, J. C.-W. (2017). Learning and Transferring Deep Joint Spectral–Spatial Features for Hyperspectral Classification. *IEEE Transactions on Geoscience and Remote Sensing*, *55*(8), 4729–4742. https://doi.org/10.1109/TGRS.2017.2698503

![image-20250901225233093](RS%20week%209.assets/image-20250901225233093.png)

**Dataset (Salinas, California):**

- Image size: **512 × 217 pixels**
- Spatial resolution: **3.7 m**
- Original spectral bands: **224**, reduced to **200** (removing noisy/low-quality bands)
- Ground truth: **16 classes** with labeled pixels (e.g., Broccoli, Celery, Grapes, Vineyard, etc.)

**Ground Truth Distribution:**

- Each class has a specific number of labeled pixels (see table).
- Example: *Grapes_untrained* has **11,271 pixels**, *Stubble* has **3,959 pixels**, etc.

**Training Strategy:**

- Different percentages of labeled ground truth pixels are used for training (e.g., **25%**).
- Example: For *Broccoli_green_weeds_1*, 502 pixels are used for training when using 25%.
- All available labeled pixels are used for **testing**, to evaluate classification performance and generalization.

Notice two important techniques:

![Capture neighborhood (spatial) properties of a pixel](RS%20week%209.assets/Capture%20neighborhood%20(spatial)%20properties%20of%20a%20pixel.png)

#### 1. **Spatial Layer**

- Purpose: Capture neighborhood (spatial) properties of a pixel.
- Method: Use a patch (e.g., 21×21 pixels) centered on the target pixel.
- Processing: Average over all spectral channels in that neighborhood.
- Input scheme:
  - **Top channel** → pixel’s spectral vector (e.g., 200 bands).
  - **Bottom channel** → neighborhood patch (spatial context).

------

#### 2. **Transfer Learning**

- Idea: CNNs trained on one hyperspectral image (with the same sensor) can transfer weights to another similar image.
- Assumption: Spatial properties are similar across images from the same sensor.
- Implementation:
  - Train convolutional layers on a different AVIRIS image.
  - Reuse these weights to initialize training on the Salinas dataset.
- Benefit: Faster convergence and reduced overfitting, similar to human learning from past experience.

| Configuration             | Testing set accuracy |
| ------------------------- | -------------------- |
| Spectral only             | 92.3%                |
| Spatial only              | 96.6%                |
| Both spectral and spatial | 98.3%                |

> Note, however, that they had to run extensive trials to find the best topology for the network-the numbers of convolutions lavers, the numbers of filters, the numbers of nodes in the hidden lavers, and so on, which indicates that the preparatory stages in using a CNN can be quite extensive.

---

Makantasis, K., Karantzalos, K., Doulamis, A., & Doulamis, N. (2015). Deep supervised learning for hyperspectral data classification through convolutional neural networks. *2015 IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 4959–4962. https://doi.org/10.1109/IGARSS.2015.7326945

![image-20250902001043287](RS%20week%209.assets/image-20250902001043287.png)

## Lecture 18. Comparing the classifiers

| Category                | Maximum likelihood classifier                                | Support vector machine                                       | CNN                                                          |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Training                | Simple process, although sets of normal distributions may need to be found for each class to get good results | Search procedures are needed to determine kernel and regularization parameters | Trial and error may be needed to find the best network topology |
| Training time           | Fast                                                         | Can be long                                                  | Can be very long; depends on the number of weights and offsets to be found |
| Classification time     | Good; depends quadratically on dimensionality                | Fast; depends linearly on dimensionality                     | Fast; depends linearly on dimensionality                     |
| Multiclass              | Is a multiclass classifier                                   | Sets of binary SVMs need to be embedded in a decision tree for multiclass operation | Is a multiclass classifier                                   |
| Hyperspectral           | Has difficulty with high - dimensional data due to the need to estimate second - order parameters | Handles hyperspectral data as it is based on linear decisions | Handles hyperspectral data as it is based on linear decisions |
| Context classification  | Is a point classifier and requires post - processors like label relaxation to embed context information | Is a point classifier                                        | Spatial context sensitivity is a key feature                 |
| Posterior probabilities | The algorithm generates class posterior probabilities prior to maximum selection | Hard classifier, so posterior probabilities are unavailable  | Surrogate posterior probabilities can be generated with the softmax operation |

> Each classifier type has its own characteristics; the analyst should choose one that does the job in other words it achieves the accuracy required without engaging in unnecessary complexity in training

Factors to be considered are:

- Ease of training, including parameter and topology selection
- Ability to handle hyperspectral data
- Multiclass capability
- Training time
- Ability to take spatial context into consideration

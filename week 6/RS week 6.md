# REMOTE SENSING IMAGE ACQUISITION, ANALYSIS AND APPLICATIONS - Week 6

Course tutor: John Richards

All original diagrams © Henry Pan. Course slides are used under fair use for educational purposes. Not for commercial use.

## Lecture 1. Fundamentals of image analysis and machine learning

| Scenario                                        | Description                                                  | Key Idea                                                     | Example / Notes                                              |
| ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **1. Point Classification**                     | Analise pixels individually based on their spectral measurements and assign labels. | Each pixel is classified independently (point-based).        | Labels represent ground covers (e.g., vegetation, soil, water). |
| **2. Context / Spatial Context Classification** | Still label individual pixels, but also consider surrounding pixels when assigning labels. | Neighboring pixels are likely of the same ground cover type. | Improves accuracy by using local context.                    |
| **3. Multisource Classification**               | Identify pixels using multiple sensors and data types (e.g., optical, radar, thermal). | Combine different data types into a single feature vector.   | Algorithms are complex; some authors simply concatenate data. |
| **4. Object-Based Classification**              | Identify larger objects rather than single pixels, especially with high-resolution imagery. | Focus on detecting objects in images.                        | Examples: buildings in cities, aircraft, vehicles in surveillance. |

```mermaid
graph LR
    A[training] --> B[classification]
    B --> C[testing]
```

The main steps of data analysis, and this module focuses on "classification"

| Classifier                                  | Description                                                  | Strengths                                                    | Limitations / Notes                                          |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Maximum Likelihood Classifier (MLC)**     | Traditional classifier in remote sensing for decades; aligns with spectral reflectance characteristics of cover types. | Easy to understand; good when spectral dimensionality is low. | Limited performance with hyperspectral data. Provides useful background concepts. |
| **Minimum Distance Classifier (MDC)**       | Very simple classification approach; measures distance to class means. | Helpful on its own; serves as an introduction to more advanced methods. | Simplicity limits accuracy in complex datasets.              |
| **Support Vector Machine (SVM)**            | More mathematically complex; separates data using optimal hyperplanes. | Widely used in past 20+ years; effective with hyperspectral datasets. | Requires more computation and parameter tuning.              |
| **Neural Networks & Deep Learning (NN/DL)** | Applied in remote sensing for ~30 years; recently boosted by convolutional neural networks and deep learning techniques. | Very powerful for complex problems; handles nonlinear relationships well. | Early versions had limited success; requires large data and computational resources. |

![Figure_1](RS%20week%206.assets/Figure_1.png)

Spectral categories are **data-driven**, while information categories are **semantics-driven**. The two do not correspond one-to-one and need to be mapped through classification algorithms

## Lecture 2. The maximum likelihood classifier

> The starting assumption for the maximum likelihood classifier is that the density of the cluster of the pixels of a given cover type in the spectral space can be described by a multi-dimensional normal distribution

Why we would say: "Even if the actual density distribution is not normal, the normal model nevertheless works well"

![Figure_2](RS%20week%206.assets/Figure_2.png)

- Even if the true data distribution is not Gaussian, many natural phenomena will exhibit approximately normal characteristics after being affected by noise superposition or the central limit theorem
- A normal model is essentially a "smoothing assumption" that can tolerate deviations and imperfections in data
- A category can be represented by a combination of multiple normal distributions. In this way, even if the true distribution is complex (multimodal, polymorphic), we can use a set of Gaussian distributions to approximate it


$$
p(\mathbf{x} \mid \omega_i) = (2\pi)^{-N/2} \vert \mathbf{C}_i \vert^{-0.5} \exp \left\{ -\frac{1}{2} (\mathbf{x} - \mathbf{m}_i)^\mathrm{T} \mathbf{C}_i^{-1} (\mathbf{x} - \mathbf{m}_i) \right\}
$$
where the m_i and C_i completely define the class distribution, and together called the **class signature**

![The explanation of maximum likelihood classifier](RS%20week%206.assets/The%20explanation%20of%20maximum%20likelihood%20classifier.png)

### Conditional probability Vs Posterior probability

Bayes theorem:
$$
\tag{1}
p(\omega_i \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \omega_i) p(\omega_i)}{p(\mathbf{x})}
$$
Decision rule:
$$
\tag{2}
\mathbf{x} \in \omega_i \quad \text{if} \quad p(\omega_i \mid \mathbf{x}) > p(\omega_j \mid \mathbf{x}) \quad \forall \, j \neq i
$$

$$
because (1)(2)
$$

$$
\tag{3}
\frac{p(\mathbf{x} \mid \omega_i) p(\omega_i)}{p(\mathbf{x})} > \frac{p(\mathbf{x} \mid \omega_j) p(\omega_j)}{p(\mathbf{x})} \quad \forall \, j \neq i
$$

Inequality with the common denominator eliminated
$$
\tag{4}
p(\mathbf{x} \mid \omega_i) p(\omega_i) > p(\mathbf{x} \mid \omega_j) p(\omega_j) \quad \forall \, j \neq i
$$
Get the simplified decision rule
$$
\tag{5}
\mathbf{x} \in \omega_i \quad \text{if} \quad p(\mathbf{x} \mid \omega_i) p(\omega_i) > p(\mathbf{x} \mid \omega_j) p(\omega_j) \quad \forall \, j \neq i
$$

$$
\text{If it is assumed that the prior probabilities of all classes are equal } (p(\omega_i) = p(\omega_j)), \text{ then it degenerates to} 
$$

$$
\begin{equation}
\tag{6}
\mathbf{x} \in \omega_i \quad \text{if} \quad p(\mathbf{x} \mid \omega_i) > p(\mathbf{x} \mid \omega_j) \quad \forall \, j \neq i
\end{equation}
$$


This is the **Maximum Likelihood Classifier** (MLC)

In the field of remote sensing, the prior probability depends on a rough assessment of the proportion of each type, or assumes that the conditional probabilities are equal when information is missing

### The derivation of Bayes theorem

Two events occurring together have a joint probability, denoted as p(x, y) . Since the joint event has no order dependence, p(x, y) = p(y, x)

The joint probability can be expressed using conditional probability:
$$
\begin{equation}
p(x, y) = p(x \mid y) p(y)
\end{equation}
$$
This means p(x, y) is the product of the conditional probability of \( x \) given \( y \) and the probability of \( y \)

Due to the order independence of the joint probability, we also have:
$$
\begin{equation}
p(x, y) = p(y, x) = p(y \mid x) p(x)
\end{equation}
$$
Equating the two expressions for \( p(x, y) \):
$$
\begin{equation}
p(x \mid y) p(y) = p(y \mid x) p(x)
\end{equation}
$$
Rearranging this equation to solve for  p(x | y)  gives Bayes' theorem:
$$
\begin{equation}
p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}
\end{equation}
$$

## Lecture 3. The maximum likelihood classifier: discriminant function and example

### A more concise form of the discriminant function

$$
% 决策规则（简化表述）
\tag{1}
\mathbf{x} \in \omega_i \quad \text{if} \quad p(\mathbf{x} \mid \omega_i) p(\omega_i) > p(\mathbf{x} \mid \omega_j) p(\omega_j) \quad \forall \, j \neq i
$$

Definition of the discriminant function:
$$
% 判别函数定义
\tag{2}
g_i(\mathbf{x}) = \ln \left\{ p(\mathbf{x} \mid \omega_i) p(\omega_i) \right\} = \ln p(\mathbf{x} \mid \omega_i) + \ln p(\omega_i)
$$
Substituting the likelihood of the multivariate normal distribution (p(x|ωᵢ)), we obtain:
$$
% 代入多元正态分布的似然（p(x|ωᵢ)）
\tag{3}
p(\mathbf{x} \mid \omega_i) = (2\pi)^{-N/2} \vert \mathbf{C}_i \vert^{-0.5} \exp \left\{ -\frac{1}{2} (\mathbf{x} - \mathbf{m}_i)^\mathrm{T} \mathbf{C}_i^{-1} (\mathbf{x} - \mathbf{m}_i) \right\}
$$
A simplified discriminant function is derived:
$$
% 推导得到简化的判别函数
\tag{4}
g_i(\mathbf{x}) = -\frac{1}{2} N \ln 2\pi - \frac{1}{2} \ln \vert \mathbf{C}_i \vert - \frac{1}{2} (\mathbf{x} - \mathbf{m}_i)^\mathrm{T} \mathbf{C}_i^{-1} (\mathbf{x} - \mathbf{m}_i) + \ln p(\omega_i)
$$
The first term contains no discriminating information and can be removed, leaving as the discriminant function for the Gaussian maximum likelihood rule
$$
\tag{5}
g_i (\mathbf{x}) = \ln p(\omega_i) - \frac{1}{2} \ln \vert \mathbf{C}_i \vert - \frac{1}{2} (\mathbf{x} - \mathbf{m}_i)^\mathrm{T} \mathbf{C}_i^{-1} (\mathbf{x} - \mathbf{m}_i)

$$
And the decision rule is:
$$
\tag{6}
\mathbf{x} \in \omega_i \quad \text{if} \quad g_i (\mathbf{x}) > g_j (\mathbf{x}) \quad \text{for all } j \neq i
$$

### some rules for MLC

> Because of the difficulty in assuring independence of the pixels, usually many more than this minimum number are selected. A practical minimum of 10N training pixels per spectral class is recommended,with as many as 100N per class if possible. That was the case for the example just considered

![Minimum Training Samples Required for the Maximum Likelihood Classifier](RS%20week%206.assets/Minimum%20Training%20Samples%20Required%20for%20the%20Maximum%20Likelihood%20Classifier.png)

Normally, we will choose 100 * N. But it will lead to a big problem with the increasing of dimensions, which called **Hughes phenomenon** or **the curse of dimensionality** often referred to in the machine learning literature

![image-20250820180803273](RS%20week%206.assets/image-20250820180803273.png)

*image source: slide 2.3.11 of this course*

This figure also illustrates that 3–4 bands are sufficient to distinguish the main classes (water, vegetation, burned areas, and urban areas). If more bands are added blindly, the number of training samples will be insufficient, which will instead lead to a decrease in classification accuracy (Hughes effect)

> Effectively, what our classifiers do is place boundaries between the classes

![Figure_1-1755696995140-6](RS%20week%206.assets/Figure_1-1755696995140-6.png)

> More complicated decision surfaces that can be obtained if we allow more than one spectral class per information class

Again, training the maximum likelihood classifier requires estimating second - order statistics of covariance matrix elements. However, problems arise when the spectral space dimensionality (i.e., number of bands) is too high (e.g., excessive bands)

However, the mean vector involves only first - order statistics, so reliable estimates are achievable even in high dimensionality (many bands). Similarly, most classifiers relying only on first - order (linear) parameters are less affected by high dimensionality and more stable

## Lecture 4. The minimum distance classifier, background material

$$
\begin{cases}
\mathbf{x} \in \text{class 1} & \text{if } \boldsymbol{w}^{\mathrm{T}}\mathbf{x} + w_{N+1} > 0 \\
\mathbf{x} \in \text{class 2} & \text{if } \boldsymbol{w}^{\mathrm{T}}\mathbf{x} + w_{N+1} < 0
\end{cases}
$$

![Linear classifier simple model and discriminant formula](RS%20week%206.assets/Linear%20classifier%20simple%20model%20and%20discriminant%20formula.png)

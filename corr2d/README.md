<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Spatial Correlation function in 2D

## Continuum fields
The autocorrelation of scalar function \\(f(\vec{x})\\) is:
$$h(\vec{r})=(f*f)(\vec{r})=\int^{\infty}_\{-\infty}\overline{f(\vec{x})}f(\vec{r}+\vec{x})dy,$$
for which
$$\hat{h}(\vec{k})=\overline{\hat{f}(\vec{k})}\hat{f}(\vec{k})=|\hat{f}(\vec{k})|^2,$$
where \\(\overline{\cdots}\\) denotes complex conjugation.

The fields we are concerned about are density \\( \rho \\), velocity \\(\vec{V}\\) and orientation \\(\vec{u}\\), coarse grained over boxes with linear size \\(l\\). To calculate the correlation functions of one field, we need calculate its Fourier function, multiply this Fourier function to its conjugate, then do a inverse Fourier function.

## Discrete particles
For the density, it's OK to do some coarse grain and calcualte correlation functions directly. However, for the velocity, the situation gets complicated since the density is not homogeneous. Recall the discrete version of correlation function for velocity:
$$C(r)=\frac{1}{c_0} \frac { \sum_ {ij}\vec{v}_i\cdot\vec{v}_j\ \delta(r-r _{ij})}{\sum _{ij} \delta (r-r _{ij})},$$
where \\(c_0 \\) is a normalization factor. The coarse-grained field is defined as:
$$\vec{v}(\vec{x}) \equiv \frac{1}{n(\vec{x})}\sum _{\vec{r}_i\sim\vec{x}} \vec{v}_i,$$
where \\(n(\vec{x})\\) is the number of particles located around \\(\vec{x}\\). Thus the correlation function for the  coarse-grained velocity is
$$C(\vec{r})=\frac{\int d\vec{x}\ n(\vec{x})n(\vec{x}+\vec{r})\ \vec{v}(\vec{x})\cdot \vec{v}(\vec{x}+\vec{r})}{\int d\vec{x}\ n(\vec{x})n(\vec{x}+\vec{r})}.$$
In this way we can get similar results when the velocity fields are obtained by coarse grained over different box sizes.
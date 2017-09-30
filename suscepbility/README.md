<script type="text/javascript"
  src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# Estimate correlation length

## Estimate correlation length from the susceptibility peak

Calculate the susceptibility peak and its location. Similar to the magnetic case, the susceptibility for the Vicsek model is defined as: 
$$\chi=L^2\langle \phi^2-\overline{\phi}^2\rangle_t,$$
where \\(\phi\\) denotes instant order parameter and \\(\overline{\cdots}\\) denotes time average. There will be a susceptibility peak as the controlling parameter approching its critical value. The location of the susceptibility peak can be regarded as the correlation length at given strength of disorder.

In the presence of disorder, the susceptibility is different from sample to sample. Two methods are proposed when do sample-averaging:

- For each sample, calculate susceptibilities with varied strength of disroder, from which we can obtain the peak and location. Then average peaks and locations over samples.
- Average susceptibilities over samples first, obtaning a sample-averaged curve for susceptibility vs. disorder strength. Then estimate the peak and location from this curve.

## Estimate correlation length from decaying of order parameters

For strength of disorder larger than its critical value, the system is in disordered state. The order parameters decay with increasing system size, and decay in a power law with exponent -1 beyond a characteristic system size. To evaulate this characteristic length, we multiply the order parameter by the system size to a given exponent, and regard the peak location of the obtained curve as the characteristic length.

## Estimate correlation length from the spatial correlation function of velocity

We can also evaluate the correlation length from the spatial correlation function of velocity fields directily, which decays exponentially in the disordered state.


## Estimate critical disorder strength and exponents from correlation length

In the disorder state, the correlation length growth rapidly with decreasing disorder strength, and goes to infity at critical disorder strength. Two types of scaling are assumed.

# Divergence of the correlation length
## KT-like scaling
$$\xi=A_{\xi}e^{b(\epsilon-\epsilon_c)^{-\nu}},$$
where \\(\nu=0.5\\) for the KT transition.

## Algebraically scaling
$$\xi=A_{\xi}(\epsilon-\epsilon_c)^{-\nu}.$$
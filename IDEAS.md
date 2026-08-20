# Gaussian Representations for Slice Extraction

Instead of decoding a CAESAR latent representation onto a grid in the requested plane, a decoder could produce a sparse collection of parameterized 2D Gaussians and rasterize or evaluate their sum on demand.

For a scientific scalar field, this is better described as a **signed anisotropic Gaussian basis** than as a probability-density Gaussian mixture model. Scientific fields may contain negative values, so component amplitudes must be signed:

$$
\hat f(u,v)=b+\sum_{i=1}^{K} a_i
\exp\left[-\frac12(\mathbf{x}-\boldsymbol\mu_i)^T
\Sigma_i^{-1}(\mathbf{x}-\boldsymbol\mu_i)\right].
$$

A decoder would predict each Gaussian's center, covariance or orientation, and amplitude from the CAESAR latent code and the requested plane.

## Comparison with a convolutional grid decoder

| Property | Convolutional grid decoder | Gaussian decoder |
|---|---|---|
| Smooth, localized structures | Good | Potentially excellent |
| Shocks and discontinuities | Better | Requires many small Gaussians |
| Turbulence and fine texture | Usually better | Can require very large $K$ |
| Arbitrary output resolution | Must upsample or rerun | Naturally continuous |
| GPU rendering | Fast and predictable | Very fast when Gaussians are sparse |
| Contour extraction | Straightforward marching squares | Still requires root finding or sampling |
| Implementation and training | Relatively simple | More complicated and less stable |

## When it could be faster

With a tile-based splatter, rendering cost is approximately proportional to the number of Gaussian-pixel overlaps. Components can be truncated at, for example, $3\sigma$, so that only nearby pixels are visited.

This can beat a dense decoder when:

- $K$ is relatively small;
- most Gaussians have limited screen extent;
- the slice contains sparse blobs, ridges, vortices, or other localized structures;
- multiple output resolutions or zoom levels are required.

It is not automatically faster. A naive evaluation costs $O(KHW)$ and would be prohibitively expensive. Efficient evaluation requires spatial binning, truncation, and a specialized GPU rasterizer. Recent 2D Gaussian image representations report extremely fast rasterization, but those results concern natural images and custom renderers rather than scientific-field accuracy ([GaussianImage](https://arxiv.org/abs/2403.08551)).

For a fixed $256^2$ slice containing dense turbulent detail, a conventional 2D CNN may still be faster because convolutions are regular and heavily optimized.

## Accuracy limitations

Gaussians are smooth. A small number can efficiently model smooth fields, but they do not naturally represent:

- discontinuities and shocks;
- extremely thin filaments;
- high-frequency oscillation;
- large nearly constant regions separated by sharp boundaries.

The decoder responds by producing many small, overlapping Gaussians. Recent Gaussian-image work explicitly identifies excessive primitive counts as a problem for high-fidelity reconstruction ([GaussianImage++](https://arxiv.org/abs/2512.19108), [SGI](https://arxiv.org/abs/2603.07789)).

For Turb-Rot-like CAESAR data, a pure Gaussian representation may work well at coarse fidelity but require many components to reproduce all fine structure.

## Contour extraction

The iso-contour of one Gaussian is an ellipse, but the iso-contour of a sum of Gaussians is not a union of those ellipses:

$$
\sum_i g_i(u,v)=\tau.
$$

Extracting a contour still requires adaptive sampling, root tracing, or a method similar to marching squares. Gaussians do provide analytic gradients and Hessians, which could make adaptive contour tracing and critical-point detection more efficient. They do not inherently preserve topology: overlapping components can introduce or remove critical points.

## A stronger variant: decode 3D Gaussians once

Rather than generating a separate Gaussian representation for every plane, decode the CAESAR latent into **3D anisotropic Gaussians**:

$$
\hat f(\mathbf{x})=\sum_i a_i
\exp\left[-\frac12(\mathbf{x}-\boldsymbol\mu_i)^T
Q_i(\mathbf{x}-\boldsymbol\mu_i)\right].
$$

Represent a requested plane as

$$
\mathbf{x}=\mathbf{o}+E\boldsymbol\xi,
\qquad
E=[\mathbf{e}_u\ \mathbf{e}_v].
$$

Restricting each 3D Gaussian to that plane produces a 2D Gaussian analytically. For component $i$:

$$
A_i=E^TQ_iE,
\qquad
\boldsymbol\xi_i=-A_i^{-1}E^TQ_i(\mathbf{o}-\boldsymbol\mu_i),
$$

and the covariance in the plane is $A_i^{-1}$. Its amplitude is reduced according to the Gaussian's distance from the plane.

This approach has several advantages:

- arbitrary slices are cheap and mutually consistent;
- components far from the plane can be discarded immediately;
- zoom and resolution are independent of the representation;
- one decoded representation supports slicing, volume rendering, and possibly isosurfaces;
- nearby slices do not fluctuate because separate decoders selected different component sets.

There is related scientific-visualization precedent for approximating volumetric data with 3D Gaussian particles, although not specifically for slicing CAESAR latents ([3D Gaussian Particle Approximation of VDB Datasets](https://arxiv.org/abs/2504.04857)).

## Practical recommendation

Test three alternatives:

1. **Convolutional plane decoder:** the simplest and likely strongest baseline.
2. **3D Gaussian decoder with analytic plane restriction:** the most interesting research direction.
3. **Hybrid representation:** a coarse grid for the background plus Gaussians for localized residual structures.

The hybrid is likely the safest approach. The grid captures broad and discontinuous behavior without requiring hundreds of broad Gaussians, while the Gaussian residual provides sparse, resolution-independent detail.

For an initial experiment, use signed Gaussians, Cholesky-parameterized covariances, $3\sigma$ support truncation, and sweep $K=64,128,256,512,1024$. Compare:

- field RMSE and maximum error;
- representation size;
- decode and rendering time;
- gradient error;
- contour distance;
- persistence-diagram error.

This experiment would determine whether the CAESAR latent contains a compact Gaussian-decodable representation rather than merely a compact grid-decodable one.

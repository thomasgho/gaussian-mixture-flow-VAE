# Gaussian Mixture VAE with Normalizing Flows

Pytorch architecture for VAE with Gaussian mixture prior and (conditional) normalizing flows. Normalizing flows implemented with pyro (conditionals can be easily taken out and replaced with non-conditional flows). Encoder/decoder is a choice of residual network or neural ODE. 

### Requirements
- Pytorch
- Pyro
- torchdiffeq

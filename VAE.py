
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import spline_autoregressive, \
    conditional_spline_autoregressive
from ODENet import *
from ResNet import *


class FlowVAE(nn.Module):

    """
    Conditional (Autoregressive) Spline Flow based VAE with ResNet/neuralODE encoder/decoder.
    Option to use a mixture of Gaussians as prior.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        num_blocks,
        num_flows,
        gauss_mix=False,
        num_gauss=14,
        tol=1e-3,
        network='resnet'):
        super(FlowVAE, self).__init__()

        self.gaussian_mixture_prior = gauss_mix

        if network == 'resnet':
            self.encoder = ResidualNet(
                input_dim,
                latent_dim,
                hidden_dim,
                num_blocks=num_blocks,
                gauss_encoder=True,
                gauss_mix=gauss_mix,
                num_gauss=num_gauss)
            self.decoder = ResidualNet(
                latent_dim,
                input_dim,
                hidden_dim,
                num_blocks=num_blocks)

        if network == 'odenet':
            self.encoder = ODEnet(
                input_dim,
                latent_dim,
                hidden_dim,
                num_blocks=num_blocks,
                gauss_encoder=True,
                gauss_mix=gauss_mix,
                num_gauss=num_gauss,
                tol=tol)
            self.decoder = ODEnet(
                latent_dim,
                input_dim,
                hidden_dim,
                num_blocks=num_blocks)

        self.flow = nn.ModuleList(
            [conditional_spline_autoregressive(latent_dim,
             context_dim=1) for _ in range(num_flows)])

    def encode(self, x):
        if self.gaussian_mixture_prior:
            (mus, logvars, weights) = self.encoder(x)
            return (mus, logvars, weights)
        else:
            (mu, logvar) = self.encoder(x)
            return (mu, logvar)

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x, context):
        if self.gaussian_mixture_prior:
            (mu, logvar, weights) = self.encode(x)
            mixture = dist.Categorical(weights.permute(1, 0))
            component = dist.Independent(
                dist.Normal(mu.permute(1, 0, 2),
                            torch.exp(logvar.permute(1, 0, 2))), 1)
            prior = dist.MixtureSameFamily(mixture, component)
        else:
            (mu, logvar) = self.encode(x)
            prior = dist.Normal(mu, logvar)

        z_dist = dist.ConditionalTransformedDistribution(
            prior,
            self.flow)

        with pyro.plate('xrd', x.shape[0]):
            z = z_dist.condition(context).sample()

        return (self.decode(z), mu, logvar, z)



def test():
    x = torch.randn((3,100))
    conditions = torch.randn((3,1))
    model = FlowVAE(
        input_dim=100,
        hidden_dim=50,
        latent_dim=10,
        num_blocks=5,
        num_flows=3,
        gauss_mix=True,
        num_gauss=5,
        network='odenet')
    x_recon, mu, logvar, z = model(x, conditions)

    print('input shape:', x.shape)
    print('output shape:', x_recon.shape)

if __name__ == "__main__":
    test()
import torch
import torch.nn as nn
from torch.autograd import Variable

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae

class VAE46(nn.Module):
    def __init__(self, nc=1, ngf=46, ndf=46, latent_variable_size=2048):
        super(VAE46, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.loss_names = ['recon_loss', 'loss']

        #for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))



        self.encoder = nn.Sequential(
            # input is n x 46 x 46
            nn.Conv2d(nc, ndf, 2, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. (ndf) x 24 x 24
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
# #             # state size. (ndf*2) x 12 x 12
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
# #             # state size. (ndf*4) x 6 x 6
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
# #              # state size. (ndf*8) x 3 x 3
            nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. (ndf*8) x 2 x 2
        )

        self.fc1 = nn.Linear(ndf*8*2*2, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*2*2, latent_variable_size)

        # decoder
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 3 x 3
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 6 x 6
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 12 x 12
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 24 x 24
            nn.ConvTranspose2d(    ngf,      nc, 2,2,1 , bias=False),
            nn.Sigmoid(),
            # state size. (nc) x 46 x 46
        )

        self.d1 = nn.Sequential(
            nn.Linear(latent_variable_size, ngf*8*2*2),
            nn.ReLU(inplace=True),
            )
        self.bn_mean = nn.BatchNorm1d(latent_variable_size)
        self.mseloss = nn.MSELoss()

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, self.ndf*8*2*2)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h = self.d1(z)
        h = h.view(-1, self.ngf*8, 2, 2)
        return self.decoder(h)

    def get_latent_var(self, x):
        x = x['image']
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def generate(self, z):
        res = self.decode(z)
        return res

    def forward(self, x):
        x = x['image']
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        return {'recon': recon, 'latent' : z, 'mu': mu, 'logvar': logvar, 'input': x}

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        print(p)
        q = torch.distributions.Normal(mu, std)
        print(q)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl


    ## from https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    def compute_loss(self, outputs, loss_trackers=None):
        recon_loss = self.gaussian_likelihood(outputs['recon'], self.log_scale, outputs['input'])

        kl = self.kl_divergence(outputs['latent'], outputs['mu'], outputs['logvar'])

        elbo = kl - recon_loss
        elbo = elbo.mean()
        # if loss_trackers:
        #     loss_trackers['loss'].add(loss.item(), len(outputs['input']))
        #     loss_trackers['recon_loss'].add(loss.item(), len(outputs['input']))

        return elbo
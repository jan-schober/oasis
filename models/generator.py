import torch
import torch.nn as nn
import torch.nn.functional as F

import models.norms as norms


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16 * ch, 16 * ch, 16 * ch, 8 * ch, 4 * ch, 2 * ch, 1 * ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        for i in range(len(self.channels) - 1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i + 1], opt))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2 ** (opt.num_res_blocks - 1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"

            '''
            Fixed Vector z for more time consistent Images, if you want to use it uncomment next line
            z = torch.tensor([[0.1940, 2.1614, -0.1721, 0.8491, -1.9244, 0.6530, -0.6494, -0.8175,
                                    0.5280, -1.2753, -1.6621, -0.3033, -0.0926, 0.1992, -1.1204, 1.8577,
                                    -0.7145, 0.6881, 0.7968, -0.0334, 1.4917, -0.5165, -0.2541, 1.4746,
                                    -0.3260, -1.1600, 2.3551, -0.6924, 0.1837, -1.1835, -1.8029, -1.5808,
                                    0.8387, 1.4192, 0.6469, 0.4253, -1.5892, 0.6223, 1.6898, -0.6648,
                                    0.9425, 0.0783, 0.0847, -0.1408, 0.3316, -0.5890, -1.0723, 0.0954,
                                    -0.3347, -0.5258, -0.8776, 0.3938, 0.1640, -0.1977, 1.0104, -1.3482,
                                    -0.3498, -0.6443, 0.4468, -0.5371, 1.2423, -0.8146, 0.2502, -0.4273]],
                                  dtype=torch.float32,
                                  device=dev)
            '''

            '''
            Next line is random z-Vector, if you want to use an fixed comment out the next line
            '''
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)

            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim=1)

        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)
            if i < self.opt.num_res_blocks - 1:
                x = self.up(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

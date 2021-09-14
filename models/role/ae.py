# import torch
# from torch.nn import Conv2d, MaxPool2d, Module, Linear, Sequential, Flatten, ELU, ConvTranspose2d


# class View(Module):
#     def __init__(self, *shape):
#         super().__init__()
#         self.shape = shape

#     def __repr__(self):
#         return f'View{self.shape}'

#     def forward(self, input):
#         batch_size = input.size(0)
#         shape = (batch_size, *self.shape)
#         out = input.view(shape)
#         return out


# class Encoder(Module):

#     def __init__(self, in_dim, num_channels, num_kernels, kernel_size, num_hidden, z_dim):
#         super(Encoder, self).__init__()

#         self.net = Sequential(
#             Conv2d(num_channels, num_kernels, kernel_size, padding=1),
#             ELU(),
#             MaxPool2d(kernel_size, stride=1, padding=1),
#             Conv2d(num_kernels, num_kernels, kernel_size, padding=1),
#             ELU(),
#             Flatten(),
#             Linear(in_dim[0] * in_dim[1] * num_kernels, num_hidden),
#             ELU(),
#             Linear(num_hidden, z_dim),
#             ELU()
#         )

#         self.training = True

#     def forward(self, x):
#         h = self.net(x)

#         return h


# class Decoder(Module):
#     def __init__(self, in_dim, num_channels, num_kernels, kernel_size, num_hidden, z_dim):
#         super(Decoder, self).__init__()

#         self.net = Sequential(
#             Linear(z_dim, num_hidden),
#             ELU(),
#             Linear(num_hidden, num_hidden),
#             ELU(),
#             Linear(num_hidden, in_dim[0] * in_dim[1] * num_kernels),
#             View(num_kernels, *in_dim),
#             ConvTranspose2d(num_kernels, num_kernels, kernel_size, padding=1),
#             ELU(),
#             ConvTranspose2d(num_kernels, num_channels, kernel_size, padding=1)
#         )

#     def forward(self, x):
#         return torch.sigmoid(self.net(x))


# class AE(Module):
#     def __init__(self, in_dim, num_channels, num_kernels, kernel_size, num_hidden, z_dim):
#         super(AE, self).__init__()
#         self.e = Encoder(in_dim, num_channels, num_kernels, kernel_size, num_hidden, z_dim)
#         self.d = Decoder(in_dim, num_channels, num_kernels, kernel_size, num_hidden, z_dim)

#     def encode(self, x):
#         return self.e(x)

#     def decode(self, z):
#         return self.d(z)

#     def forward(self, x):
#         z = self.encode(x)
#         x = self.decode(z)

#         return x

# 3p
import torch
import torch.nn as nn
import torch.nn.functional as F
import roma
from diffusion_net import DiffusionNet

FC_SIZE_1 = 256
FC_SIZE_2 = 128
class MLP(nn.Module):
    def __init__(self,dim_out, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(MLP, self).__init__()
        self.dim_out = dim_out
        self.fc1 = nn.Linear(dim_out, 512, device=device)
        self.fc2 = nn.Linear(512, 256, device=device)
        self.fc3 = nn.Linear(256, 12, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.shape[2] >= 512:
            resid = x[:,:,:512]
        else:
            pad_left = (512 - x.shape[2])//2
            pad_right = 512 - x.shape[2] - pad_left
            resid = F.pad(x, (pad_left, pad_right, 0, 0, 0, 0) , "constant", 0)
        x = resid + self.relu(self.fc1(x)) # x + F.pad(x,  (int(FC_SIZE/2) - self.neighborhood_size,int(FC_SIZE/2) - self.neighborhood_size), "constant", 0) + self.relu(self.fc1(x))
        x = x[:,:,:256] + self.relu(self.fc2(x))
        x = x[:,:,:12] + self.fc3(x)
        out = x
        return out
    
class PrismDecoder(torch.nn.Module):
    def __init__(self, dim_in=1024, dim_out=512, n_width=256, n_block=4):
        # , pairwise_dot=True, dropout=False, dot_linear_complex=True, neig=128):
        super().__init__()

        self.diffusion_net = DiffusionNet(C_in=dim_in, C_out=dim_out, C_width=n_width, N_block=n_block)
                                        #   , pairwise_dot=pairwise_dot, dropout=dropout,
                                        #   dot_linear_complex=dot_linear_complex, neig=neig)

        # self.mlp_refine = MLP(dim_out)
        
        self.mlp_refine = nn.Sequential(
            nn.Linear(dim_out, dim_out),
            nn.ReLU(),
            nn.Linear(dim_out, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, batch):
        # original prism
        verts = batch.pos.reshape(-1, 3)
        faces = batch.face.t()
        prism_base = verts[faces]  # (n_faces, 3, 3)
        bs, _, _ = batch.pos.shape #

        # forward through diffusion net
        batch = self.diffusion_net(batch)  # (bs, n_verts, dim)

        # features per face
        x = batch.x.unsqueeze(0)
        x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
        faces_gather = faces.unsqueeze(0).unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
        xf = torch.gather(x_gather, 1, faces_gather)
        features = torch.mean(xf, dim=-1)  # (bs, n_faces, dim)

        # refine features with mlp
        features = self.mlp_refine(features)  # (bs, n_faces, 12)

        # get the translation and rotation
        rotations = features[:, :, :9].reshape(-1, 3, 3)
        rotations = roma.special_procrustes(rotations)  # (n_faces, 3, 3)
        translations = features[:, :, 9:].reshape(-1, 3)  # (n_faces, 3)

        # transform the prism
        transformed_prism = (prism_base @ rotations) + translations[:, None]

        # prism to vertices
        features = self.prism_to_vertices(transformed_prism, faces, verts)

        batch.features = features.reshape(bs, -1, 3)
        batch.transformed_prism = transformed_prism
        batch.rotations = rotations
        return batch

    def prism_to_vertices(self, prism, faces, verts):
        # initialize the transformed features tensor
        N = verts.shape[0]
        d = prism.shape[-1]
        device = prism.device
        features = torch.zeros((N, d), device=device)

        # scatter the features in K onto L using the indices in F
        features.scatter_add_(0, faces[:, :, None].repeat(1, 1, d).reshape(-1, d), prism.reshape(-1, d))

        # divide each row in the transformed features tensor by the number of faces that the corresponding vertex appears in
        num_faces_per_vertex = torch.zeros(N, dtype=torch.float32, device=device)
        num_faces_per_vertex.index_add_(0, faces.reshape(-1), torch.ones(faces.shape[0] * 3, device=device))
        features /= num_faces_per_vertex.unsqueeze(1).clamp(min=1)

        return features

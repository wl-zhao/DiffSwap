import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arcface_torch.backbones import get_model
from src.arcface_torch.utils.utils_config import get_config

from einops import rearrange
from ldm.modules.attention import CrossAttention

from ldm.models.diffusion.ddpm import disabled_train


# including l_eye r_eye nose mouse
class FaceEmbedder(nn.Module):
    def __init__(self, lmk_dim=128, keys=None, pair=False, comb_mode='concat', merge_eyes=False, \
            attention=False, face_model='r50', face_dataset='glint360k', affine_crop=False, use_blur=False):
        super().__init__()
        self.pair = pair
        cfg = get_config(f'src/arcface_torch/configs/{face_dataset}_{face_model}.py')
        self.face_model = get_model(cfg.network, dropout=0.0, 
                fp16=cfg.fp16, num_features=cfg.embedding_size)
        ckpt_path = f'checkpoints/{face_dataset}_{face_model}.pth'
        a, b = self.face_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
        print('loading face model:', a, b)
        print('build model:', face_dataset, face_model, ckpt_path)
        
        self.landmark_encoder = nn.Sequential(
            nn.Linear(68 * 2, lmk_dim * 4),
            nn.GELU(),
            nn.Linear(lmk_dim * 4, lmk_dim)
        )
        assert 'image' in keys and 'landmark' in keys
        self.swap = False
        self.comb_mode = comb_mode
        if lmk_dim != 512 and self.comb_mode == 'stack':
            self.id_fc = nn.Sequential(
                nn.Linear(512, 2 * lmk_dim),
                nn.GELU(),
                nn.Linear(2 * lmk_dim, lmk_dim)
            )
        else:
            self.id_fc = None

        self.organ_fc = nn.Sequential(
            nn.Linear(3, lmk_dim),
            nn.GELU(),
            nn.Linear(lmk_dim, lmk_dim)
        )
        
        if attention:
            self.organ_norm = nn.LayerNorm(lmk_dim)
            self.organ_attention = CrossAttention(
                query_dim=lmk_dim,
                context_dim=lmk_dim,
                dim_head=32,
                heads=4,
            )
            print('[FaceEmbedder]: building organ attention')
        else:
            self.organ_attention = None
        self.organ_keys = ['l_eye', 'r_eye', 'mouth', 'nose']
        self.merge_eyes = merge_eyes
        self.affine_crop = affine_crop
        self.face_model.eval()
        self.face_model.train = disabled_train
        self.use_blur = use_blur

    def encode_face(self, image):
        # image: (b c h w)
        id_feat = self.face_model(image)
        return id_feat

    def extract_organ_feats(self, z, mask_organ):
        if self.merge_eyes:
            mask_organ_ = mask_organ[:, 1:]
            mask_organ_[:, 0] = torch.logical_or(mask_organ[:, 0], mask_organ[:, 1])
            mask_organ = mask_organ_
        else:
            pass

        h, w = z.shape[-2:]
        mask_organ = F.interpolate(mask_organ.float(), (h, w), mode='nearest')
        sum1 = torch.einsum('bchw,bkhw->bkc', z, mask_organ)
        sum2 = mask_organ.sum(dim=(-1, -2))[..., None]
        return sum1 / (sum2 + 1e-6)

    def affine_crop_face(self, img, affine_theta):
        grid = F.affine_grid(affine_theta, size=(img.size(0), 3, 112, 112))
        img = F.grid_sample(img, grid)
        return img

    def forward(self, cond, swap=False):
        swap = swap or self.swap

        image, landmark = cond['image'], cond['landmark']
        z = cond['z']

        if swap:
            assert 'image_src' in cond
            image = cond['image_src']
            z = cond['z_src']
            mask_organ = cond['mask_organ_src']
            affine_theta = cond['affine_theta_src']
        else:
            mask_organ = cond['mask_organ']
            affine_theta = cond['affine_theta']


        # extract organ features using z
        organ_feats = self.extract_organ_feats(z, mask_organ)
        cond['organ_feat'] = organ_feats
        organ_feats = self.organ_fc(organ_feats)
        if self.organ_attention is not None:
            organ_feats = organ_feats + self.organ_attention(self.organ_norm(organ_feats))

        B = image.size(0)
        
        with torch.no_grad():
            image = rearrange(image, 'b h w c -> b c h w')
            if self.affine_crop:
                image = self.affine_crop_face(image, affine_theta)
            else:
                image = F.interpolate(image, (112, 112), mode='bicubic')
            id_feat = self.encode_face(image)

        cond['id_feat'] = id_feat

        if self.id_fc:
            id_feat = self.id_fc(id_feat)
            
        lmk_feat = self.landmark_encoder(landmark.reshape(B, -1))
        # print(self.landmark_encoder[0].weight.sum())
        if self.comb_mode == 'concat':
            raise NotImplementedError()
            out = torch.cat([id_feat, lmk_feat], dim=1)[:, None]
        elif self.comb_mode == 'stack':
            out = torch.stack([lmk_feat, id_feat], dim=1)
            out = torch.cat([out, organ_feats], dim=1)
        else:
            raise NotImplementedError()
        # if self.training:
        
        if self.use_blur:
            return {
                'c_concat': cond['z_blur'],
                'c_crossattn': out
            }
        else:
            return out
    
    def parameters(self):
        params = list(self.landmark_encoder.parameters())
        if self.id_fc:
            params += list(self.id_fc.parameters())
        params += list(self.organ_fc.parameters())
        if self.organ_attention:
            params += list(self.organ_attention.parameters())
            params += list(self.organ_norm.parameters())
        return params


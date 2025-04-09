"""
Basic OSTrack model.
"""
from copy import deepcopy
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.artrackv2_seq.vit import vit_base_patch16_224, vit_large_patch16_224, vit_tiny_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.mask_decoder import build_maskdecoder

from lib.models.layers.head import build_decoder, MLP, DropPathAllocator

import time


class ARTrackV2Seq(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer,
             #    cross_2_decoder,
              #   score_mlp,
                 ):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
    #    self.score_mlp = score_mlp

        self.identity = torch.nn.Parameter(torch.zeros(1, 6, 192))
       # self.x_identity = torch.nn.Parameter(torch.zeros(1, 1, 192))
        self.identity = trunc_normal_(self.identity, std=.02)
     #   self.x_identity = trunc_normal_(self.x_identity, std=.02)

   #     self.cross_2_decoder = cross_2_decoder

    def forward(self, template: torch.Tensor,
               # dz_feat: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_input=None,
                head_type=None,
                stage=None,
                search_feature=None,
                target_in_search_img=None,
                gt_bboxes=None,
                ):
        template_0 = template
        out, z_0_feat, z_1_feat, x_feat, score_feat = self.backbone(z_0=template_0, x=search, identity=self.identity, seqs_input=seq_input,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,)

        return out
        

class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg, hidden_dim):
    return MlpScoreDecoder(
        in_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        bn=False
    )

def build_artrackv2_seq(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = "/home/baiyifan/code/4template/pretrained_models/"
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        print("i use vit_large")
        backbone = vit_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    cross_2_decoder = build_maskdecoder(cfg, hidden_dim)

    drop_path = cfg.MODEL.DROP_PATH
    drop_path_allocator = DropPathAllocator(drop_path)
    num_heads = cfg.MODEL.NUM_HEADS
    mlp_ratio = cfg.MODEL.MLP_RATIO
    qkv_bias = cfg.MODEL.QKV_BIAS
    drop_rate = cfg.MODEL.DROP_RATE
    attn_drop = cfg.MODEL.ATTN_DROP
    score_mlp = build_score_decoder(cfg, hidden_dim)

    model = ARTrackV2Seq(
        backbone,
   #     cross_2_decoder,
   #     score_mlp,
    )
    load_from = cfg.MODEL.PRETRAIN_PTH
    checkpoint = torch.load(load_from, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    print('Load pretrained model from: ' + load_from)
    with torch.no_grad():
            # 获取position_embeddings的权重（形状：[4, in_channel]）
        source_weights = model.backbone.position_embeddings.weight.data
            
            # 在第一个维度（num_embeddings方向）重复prenum次（新形状：[prenum*4, in_channel]）
        repeated_weights = source_weights.repeat(3, 1)
            
            # 将重复后的权重赋给prev_position_embeddings
        model.backbone.prev_position_embeddings.weight.data.copy_(repeated_weights)
    if 'sequence' in cfg.MODEL.PRETRAIN_FILE and training:
        print("i change myself")
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

import argparse
import torch
import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
import _init_paths
from lib.utils.merge import merge_template_search
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
# import lib.models.HiT.levit_utils as utils


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='artrackv2_seq',
                        help='training script name')
    parser.add_argument('--config', type=str, default='artrackv2_seq_256_got', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, images_list, xz, run_box_head, run_cls_head, bs):
    """Compute FLOPs, Params, and Speed"""
    custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    # # backbone
    macs1, params1 = profile(model, inputs=(images_list, None, "backbone", False, False),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('backbone macs is ', macs)
    print('backbone params is ', params)
    #head
    macs2, params2 = profile(model, inputs=(None, xz, "head", True, True),
                             custom_ops=custom_ops, verbose=False)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('head macs is ', macs)
    print('head params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2, params1 + params2], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 50
    T_t = 500
    print("testing speed ...")
    with torch.no_grad():
        # # overall
        for i in range(T_w):
            _ = model(images_list, None, "backbone", run_box_head, run_cls_head)
            _ = model(None, xz, "head", run_box_head, run_cls_head)
        min_latency = float('inf')
        for i in range(T_t):
            start = time.time()
            _ = model(images_list, None, "backbone", run_box_head, run_cls_head)
            _ = model(None, xz, "head", run_box_head, run_cls_head) 
            end = time.time()
            cur_latency = (end - start) / bs
            min_latency = min(min_latency, cur_latency)
            
        latency_ms = min_latency * 1000
        fps = 1000 / latency_ms
        print("The minimum overall latency is %.2f ms" % latency_ms)
        print("FPS: %.2f" % fps)

def evaluate_mixformer(model, template, online_template, search, bs=1):
    """Compute FLOPs, Params, and Speed"""
    # (template, online_template, search, softmax, remove_rate_cur_epoch)
    macs1, params1 = profile(model, inputs=(template, online_template, search, False, 1.0),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template, online_template, search, False, 1.0)
        
        min_latency = float('inf')
        for i in range(T_t):
            start = time.time()
            _ = model(template, online_template, search, False, 1.0)
            end = time.time()
            cur_latency = (end - start) / bs
            min_latency = min(min_latency, cur_latency)
            
        latency_ms = min_latency * 1000
        fps = 1000 / latency_ms
        print("The minimum overall latency is %.2f ms" % latency_ms)
        print("FPS: %.2f" % fps)


def evaluate_artrack(model, template, dz_feat, search, seqs_input, bs=1):
    """Compute FLOPs, Params, and Speed"""
    # (template, dz_feat, search, _, _, _, seqs_input)
    macs1, params1 = profile(model, inputs=(template, dz_feat, search, None, None, False, seqs_input),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        # for i in range(T_w):
        #     _ = model(template, dz_feat, search, None, None, False, seqs_input)
            
        # min_latency = float('inf')
        # for i in range(T_t):
        #     start = time.time()
        #     _ = model(template, dz_feat, search, None, None, False, seqs_input)
        #     end = time.time()
        #     cur_latency = (end - start) / bs
        #     min_latency = min(min_latency, cur_latency)
            
        # latency_ms = min_latency * 1000
        # fps = 1000 / latency_ms
        # print("The minimum overall latency is %.2f ms" % latency_ms)
        # print("FPS: %.2f" % fps)
        # overall
        for i in range(T_w):
            _ = model(template, dz_feat, search, None, None, False, seqs_input)
        start = time.time()
        for i in range(T_t):
            _ = model(template, dz_feat, search, None, None, False, seqs_input)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))

def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":

    args = parse_args()
    '''update cfg'''
    yaml_fname = prj_path + '/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    # h_dim = cfg.MODEL.HIDDEN_DIM
    '''import vt network module'''
    # model_module = importlib.import_module('lib.models.mixformer2_vit')
    model_module = importlib.import_module('lib.models.artrackv2_seq')
    if args.script == "mixformer2_vit_stu":
        model_constructor = model_module.build_mixformer2_vit_stu
        model = model_constructor(cfg) # 要注意模型是否默认Train=False
        # get the template and search
        # change the head's params' device to CPU
        template = get_data(bs, z_sz)
        online_template = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # images_list = [template, online_template, search]
        evaluate_mixformer(model, template, online_template, search, bs=bs)
    elif args.script == "artrackv2_seq":
        # torch.backends.cudnn.benchmark = False
        model_constructor = model_module.build_artrackv2_seq
        model = model_constructor(cfg, training=False) # 要注意模型是否默认Train=False
        # get the template and search
        # change the head's params' device to CPU
        # template = get_data(bs, z_sz)
        template = torch.randn(bs, 1, 3, z_sz, z_sz)
        dz_feat = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # seqs_input = torch.randn(bs, 28) # 4*7坐标序列
        seqs_input = torch.randint(1, 401, (bs, 28))
        evaluate_artrack(model, template, dz_feat, search, seqs_input, bs=bs)
    elif args.script in ["vtm1", "vtm2"]:
        model_constructor = model_module.build_vtm
        model = model_constructor(cfg)
        # merge conv+bn for levit
        # if "LeViT" in cfg.MODEL.BACKBONE.TYPE:
        #     # merge conv+bn to one operator
        #     utils.replace_batchnorm(model.backbone.body)
        # get the template and search
        template1 = get_data(bs, z_sz)
        template2 = get_data(bs, z_sz)
        search = get_data(bs, x_sz)
        # transfer to device
        # model = model.to(device)
        # template1 = template1.to(device)
        # template2 = template2.to(device)
        # search = search.to(device)
        # forward template and search
        images_list = [search, template1, template2]
        xz = model.forward_backbone(images_list)
        # evaluate the model properties
        evaluate(model, images_list, xz, run_box_head=True, run_cls_head=True, bs=bs)

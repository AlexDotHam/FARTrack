class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = "/home/baiyifan/liangshiyi/AR2_parallel/2stage_large/"    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/baiyifan/code/2stage/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/baiyifan/code/2stage/pretrained_networks'
        self.lasot_dir = '/data3/wangqiang/LaSOTBenchmark'
        self.got10k_dir = '/data3/wangqiang/GOT-10K/train'
        self.got10k_val_dir = '/data3/wangqiang/GOT-10K/val'
        self.lasot_lmdb_dir = '/home/baiyifan/code/2stage/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/baiyifan/code/2stage/data/got10k_lmdb'
        self.trackingnet_dir = '/data3/wangqiang/TrackingNet/track'
        self.trackingnet_lmdb_dir = '/home/baiyifan/code/2stage/data/trackingnet_lmdb'
        self.coco_dir = '/data3/wangqiang/coco2017'
        self.coco_lmdb_dir = '/home/baiyifan/code/2stage/data/coco_lmdb'
        self.sav_dir = '/data2/SAV_mannual/translate'
        self.vasttrack_dir = "/data3/VastTrack/train"
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/baiyifan/code/2stage/data/vid'
        self.imagenet_lmdb_dir = '/home/baiyifan/code/2stage/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

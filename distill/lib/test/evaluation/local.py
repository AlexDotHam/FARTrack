from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/wangguijie/code/Siamese-ResNet-track/data/got10k_lmdb'
    settings.got10k_path = '/data3/wangqiang/GOT-10K'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/wangguijie/code/Siamese-ResNet-track/data/itb'
    settings.lasot_extension_subset_path = '/data3/wangqiang/LaSOT_extension_subset'
    settings.lasot_lmdb_path = '/home/baiyifan/code/2stage/data/lasot_lmdb'
    settings.lasot_path = '/data3/wangqiang/LaSOTBenchmark'
    settings.network_path = '/data5/artrack_tiny_224_distill/'   # Where tracking networks are stored.
    settings.nfs_path = '/home/wangguijie/code/Siamese-ResNet-track/data/nfs'
    settings.otb_path = '/home/wangguijie/code/Siamese-ResNet-track/data/otb'
    settings.prj_dir = '/home/baiyifan/code/distill/'
    settings.result_plot_path = '/data5/artrack_tiny_224_distill/'
    settings.results_path = '/data5/artrack_tiny_224_distill/'    # Where to store tracking results
    settings.save_dir = '/data5/artrack_tiny_224_distill/'
    settings.segmentation_path = '/home/wangguijie/code/Siamese-ResNet-track/output/test/segmentation_results'
    settings.tc128_path = '/home/wangguijie/code/Siamese-ResNet-track/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/wangguijie/code/Siamese-ResNet-track/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data3/wangqiang/TrackingNet/track'
    settings.uav_path = '/home/wangguijie/code/Siamese-ResNet-track/data/uav'
    settings.vot18_path = '/home/wangguijie/code/Siamese-ResNet-track/data/vot2018'
    settings.vot22_path = '/home/wangguijie/code/Siamese-ResNet-track/data/vot2022'
    settings.vot_path = '/home/wangguijie/code/Siamese-ResNet-track/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings


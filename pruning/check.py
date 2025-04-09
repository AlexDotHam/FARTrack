with open('/home/baiyifan/tiny/4template/lib/models/artrackv2_seq/base_backbone.py', 'rb') as f:
    content = f.read()
    print(content[2590:2610])  # 查看出错位置附近的字节
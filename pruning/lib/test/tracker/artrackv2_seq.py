import math

from lib.models.artrackv2_seq import build_artrackv2_seq
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os
import numpy as np
import shutil
from PIL import Image

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import re

std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.485, 0.456, 0.406])
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def apply_mask_and_generate_mask_image(image_path, mask_tensor, output_image_path, output_mask_path=None):
    # 读取图像并转换为RGBA模式以处理透明度
    overlay_alpha = 128
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    data = np.array(img)

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    overlay_data = np.array(overlay)

    # 将mask tensor转换为numpy数组并重塑为7x7
    mask_np = mask_tensor.cpu().numpy().reshape(7, 7)

    # 定义patch大小和图像尺寸
    patch_size = 16
    image_size = 112

    # 应用mask到图像
    patch_size = 16
    for i in range(7):
        for j in range(7):
            if mask_np[i, j] == 0:
                # 计算坐标范围
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size

                # 设置半透明白色（RGB=白色，alpha=指定值）
                overlay_data[y_start:y_end, x_start:x_end] = [255, 255, 255, overlay_alpha]

    # 合并图层
    overlay = Image.fromarray(overlay_data, 'RGBA')
    composite = Image.alpha_composite(img, overlay)

    # 保存处理后的图像
    overlay = Image.fromarray(overlay_data, 'RGBA')
    composite.save(output_image_path)

    # 生成mask图（被遮蔽区域为白色，其余为黑色）
    # mask_array = np.zeros((image_size, image_size), dtype=np.uint8)
    # for i in range(7):
    #     for j in range(7):
    #         if mask_np[i, j] == 0:
    #             mask_array[
    #             i * patch_size: (i + 1) * patch_size,
    #             j * patch_size: (j + 1) * patch_size
    #             ] = 255
    # mask_img = Image.fromarray(mask_array, mode='L')
    # mask_img.save(output_mask_path)

class ARTrackV2Seq(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ARTrackV2Seq, self).__init__(params)
        network = build_artrackv2_seq(params.cfg, training=False)
     #   checkpoint = torch.load(self.params.checkpoint, map_location='cpu')
      #  new_checkpoints = checkpoint['net'].copy()
      #  for key in checkpoint['net'].keys():
      #      pattern = re.compile(r'(\w+\.)*(norm\d+)(\.\w+)*')
      #      if 'norm' in key and 'masknorm' not in key and 'norm.' not in key:
            # 找到对应的 masknorm 层的键
      #          match = pattern.match(key)
      #          if match:
      #              norm_part = match.group(2)
                # 构建对应的 masknorm 层的键
      #              masknorm_key = key.replace(norm_part, f'mask{norm_part}.norm')
                # 将 norm 层的权重复制到对应的 masknorm 层
          
      #          new_checkpoints[masknorm_key] = checkpoint['net'][key]
    
       # checkpoint["net"] = new_checkpoints
       # network.load_state_dict(checkpoint['net'], strict=True)

        self.cfg = params.cfg
        self.bins = params.cfg.MODEL.BINS
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.dz_feat = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.store_result = None
        self.prenum = params.cfg.MODEL.PRENUM
        self.range = params.cfg.MODEL.RANGE
        self.x_feat = None
        self.allnum = 0

    def template_update(self, new_z, strategy_type=0):
    
  #  Args:
   #     new_z: 新的模板
   #     strategy_type: 选择更新策略的类型:
   #         0: 固定间隔顺序更新 - 每隔固定帧数按顺序更新一个模板 # 20 74.6 10 74.5 5 74.0 30 73.8
  #          1: 指数衰减更新 - 不同位置的模板以指数递增的间隔进行更新 # 10 73.2
     #       2: 优先级队列更新 - 根据模板年龄和位置综合确定更新优先级 # 10 74.1
      #      3: 概率随机更新 - 根据模板位置设置不同概率进行随机更新
     #       4: 多频率轮转更新 - 不同位置模板以不同但固定的频率更新
        if strategy_type == 0:
        # 固定间隔顺序更新
            interval = 20  # 可调整的帧间隔
            if self.frame_id % interval == 0:
                update_idx = (self.frame_id // interval) % (self.num_template - 2) + 1
                self.z_dict1[update_idx] = new_z
            
        elif strategy_type == 1:
        # 指数衰减更新
            for i in range(1, self.num_template):
                update_freq = 2 ** (i-1) * 10  # 指数增长的更新间隔
                if self.frame_id % update_freq == 0:
                    self.z_dict1[i] = new_z
                
        elif strategy_type == 2:
        # 优先级队列更新
            if not hasattr(self, 'template_ages'):
                self.template_ages = [0] * self.num_template
            
            self.template_ages = [age + 1 for age in self.template_ages]
            self.template_ages[0] = 0  # 确保第一个模板不更新
        
        # 计算优先级 (跳过第一个模板)
            priorities = [age * (1 + 0.1 * i) for i, age in enumerate(self.template_ages[1:])]
        
            if self.frame_id % 15 == 0:  # 每10帧检查一次
                update_idx = np.argmax(priorities) + 1  # +1 因为跳过了第一个模板
                self.z_dict1[update_idx] = new_z
                self.template_ages[update_idx] = 0
            
        elif strategy_type == 3:
        # 概率随机更新
            if self.frame_id % 20 == 0:  # 每20帧更新一次
            # 生成更新概率，越近的模板概率越高，跳过第一个模板
                probs = np.array([(self.num_template - i) / self.num_template 
                             for i in range(1, self.num_template - 1)])
                probs = probs / probs.sum()
            
            # 随机选择一个模板位置进行更新（跳过第一个模板）
                update_idx = np.random.choice(range(1, self.num_template), p=probs)
                self.z_dict1[update_idx] = new_z
            
        elif strategy_type == 4:
        # 多频率轮转更新
            if not hasattr(self, 'update_schedule'):
            # 初始化更新计划，跳过第一个模板
                self.update_schedule = []
                base_freqs = [5, 10, 20, 40]  # 不同的更新频率
                for i, freq in enumerate(base_freqs, 1):  # 从索引1开始
                    self.update_schedule.append((i, freq))
        
            for template_idx, freq in self.update_schedule:
                if self.frame_id % freq == 0:
                    self.z_dict1[template_idx] = new_z
        elif strategy_type == 5:
        # **智能阈值更新**
        # 根据新模板与现有模板的相似性，仅在相似性低于某个阈值时更新
            similarity_threshold = 0.7  # 设置相似性阈值
            if not hasattr(self, 'template_similarities'):
            # 初始化模板相似性分数
                self.template_similarities = [1.0] * self.num_template

        # 每隔固定间隔检查
            interval = 15
            if self.frame_id % interval == 0:
            # 计算每个模板的相似性
                similarities = [np.dot(self.z_dict1[i], new_z) / 
                            (np.linalg.norm(self.z_dict1[i]) * np.linalg.norm(new_z)) 
                            for i in range(1, self.num_template)]
            
            # 找到相似性低于阈值的模板进行更新
                update_idx = np.argmin(similarities) + 1
                self.z_dict1[update_idx] = new_z
          #      self.template_similarities[update_idx-1] = similarities[update_idx-1]
              #  for i, sim in enumerate(similarities, start=1):
              #      if sim < similarity_threshold:
              #          self.z_dict1[i] = new_z
              #          self.template_similarities[i] = sim  # 更新相似性分数
        else:
            raise ValueError("不支持的更新策略类型")

    def template_update_sampling(self, new_z, sampling_method="linear", mask=None):
  #  """
 #   更新模板，支持多种采样方法。
    
 #   Args:
   #     new_z: 新的模板。
   #     sampling_method: 采样方法，支持以下选项：
     #       - "linear": 线性均匀采样。 # 95 762
    #        - "exponential": 指数衰减采样（越近越密集）。0.5 75.3 0.25 74.8 0.75 77 0.80 76.8 0.7 77.1
      #      - "logarithmic": 对数递增采样（越早越密集）。76.1
      #      - "random": 随机采样（第0帧和最新帧始终保留）。
      #      - "fixed_weight": 固定间隔递减权重采样。
  #  """
    # 初始化存储所有帧的模板
        if not hasattr(self, 'stored_templates'):
            self.stored_templates = []  # 用于存储每帧的new_z
            self.stored_templates.append(self.z_dict1[0])  # 第0帧初始化到模板0
            self.store_mask = []
            self.store_mask_50 = []
            self.store_mask_75 = []
            self.store_mask_90 = []
          #  self.store_mask.append(mask[:, 49*4:49*5])

    # 从第二帧开始存储每一个新的new_z
        if self.frame_id >= 1:
            self.stored_templates.append(new_z)

            self.store_mask.append(mask[0])
            self.store_mask_50.append(mask[1])
            self.store_mask_75.append(mask[2])
            self.store_mask_90.append(mask[3])

    # 当前帧数
        current_frame_count = self.frame_id + 1  # 从第0帧开始计数
        num_templates = self.num_template  # 模板数量
        mask_temp = torch.ones(1, 445) > 0

    # 如果帧数小于模板数量，直接按顺序填充模板
        if current_frame_count < num_templates:
            mode = 'mode1'  # 可以改为'mode2'切换模式
    
            for template_pos in range(num_templates):
        # 模式1：渐进式填充（0,0,0,1,2）
                if mode == 'mode1':
                    shift = num_templates - current_frame_count
                    template_idx = max(0, template_pos - shift)
        
        # 模式2：快速过渡到新模板（0,1,1,1,1）
                elif mode == 'mode2':
                    template_idx = min(template_pos, current_frame_count - 1)
        
        # 处理索引越界的情况（安全保护）
                template_idx = min(template_idx, len(self.stored_templates)-1)
        
                self.z_dict1[template_pos] = self.stored_templates[template_idx]
                if template_idx < len(self.store_mask):
                  #  print(template_idx, len(self.store_mask))
                    mask_temp[:, 49*template_pos:49*(template_pos+1)] = self.store_mask[template_idx]
                    #print(mask_temp)
                else:
                	mask_extra = torch.ones([1, 49]) > 0
                	mask_temp[:, 49*template_pos:49*(template_pos+1)] = mask_extra
               
                #print(template_pos, template_idx)
            mask_temp = mask_temp.unsqueeze(-1).expand(-1, -1, 445).permute(0, 2, 1)
            #print(mask_temp)
            self.mask = mask_temp
            return
    # 根据采样方法计算采样点索引
        if sampling_method == "linear":
        # 线性均匀采样
            step = (current_frame_count - 1) / (num_templates - 1)
            sampled_indices = [int(i * step) for i in range(num_templates - 1)]
            sampled_indices.append(current_frame_count - 1)

        elif sampling_method == "exponential":
        # 指数衰减采样：越近越密集
            sampled_indices = [0]  # 第0帧始终保留
            for i in range(1, num_templates - 1):
                index = int((current_frame_count - 1) * (1 - 0.7 ** i))
                sampled_indices.append(index)
            sampled_indices.append(current_frame_count - 1)  # 保留最新帧

        elif sampling_method == "logarithmic":
        # 对数递增采样：越早越密集
            sampled_indices = [0]  # 第0帧始终保留
            for i in range(1, num_templates - 1):
                index = int((current_frame_count - 1) * (i / (num_templates - 1)) ** 0.5)
                sampled_indices.append(index)
            sampled_indices.append(current_frame_count - 1)  # 保留最新帧

        elif sampling_method == "random":
        # 随机采样
            sampled_indices = [0, current_frame_count - 1]  # 第0帧和最新帧
            if num_templates > 2:
                additional_indices = sorted(
                    random.sample(range(1, current_frame_count - 1), num_templates - 2)
                )
                sampled_indices = [0] + additional_indices + [current_frame_count - 1]

        elif sampling_method == "fixed_weight":
        # 固定间隔递减权重采样
            weights = [(1 / (i + 1)) for i in range(current_frame_count)]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            sampled_indices = sorted(
                random.choices(range(current_frame_count), weights=probabilities, k=num_templates - 1)
            )
            sampled_indices[0] = 0  # 确保第0帧
            sampled_indices[-1] = current_frame_count - 1  # 保留最新帧

        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    # 更新模板
        for i, idx in enumerate(sampled_indices):
            self.z_dict1[i] = self.stored_templates[idx]
            if idx < len(self.store_mask):
                mask_temp[:, 49*i:49*(i+1)] = self.store_mask[idx]
            else:
                mask_extra = torch.ones([1, 49]) > 0
                mask_temp[:, 49*i:49*(i+1)] = mask_extra
        mask_temp = mask_temp.unsqueeze(-1).expand(-1, -1, 445).permute(0, 2, 1)
        self.mask = mask_temp

    def initialize(self, image, info: dict, name:str):
        # forward the template once
        self.root = f"/data5/got10k_template_mask/"
        self.name = name
        self.x_feat = None
        self.update_ = False

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)  # output_sz=self.params.template_size
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # initialize dynamic template as template in first frame
            self.z_dict1 = [template.tensors] * self.num_template

        self.box_mask_z = None
        self.mask = torch.ones([1, 445, 445]) > 0
        self.mask = self.mask.cuda()

        # save states
        self.state = info['init_bbox']
        self.store_result = [info['init_bbox'].copy()]
        for i in range(self.prenum - 1):
            self.store_result.append(info['init_bbox'].copy())
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        path = self.root + self.name
        if self.frame_id==0:
            self.allnum+=1
            path = self.root + self.name
            if os.path.exists(path):
                shutil.rmtree(path)
            os.mkdir(path)
            os.mkdir(path+"/template_whole/")


        H, W, _ = image.shape
        self.frame_id += 1

        if self.frame_id == 1:
            name = "template_0" + ".png"
            template = self.z_dict1[0].cpu().reshape(3, 112, 112).permute(1, 2, 0).numpy()
            template = np.clip(template * std + mean, 0, 1) * 255
            im = Image.fromarray(np.uint8((template)))
            im.save(path + "/" + name)
        else:
            name = "template_" + str(self.frame_id - 1) + ".png"
            template = self.z_dict1[-1].cpu().reshape(3, 112, 112).permute(1, 2, 0).numpy()
            template = np.clip(template * std + mean, 0, 1) * 255
            im = Image.fromarray(np.uint8((template)))
            im.save(path + "/" + name)

            idx = self.frame_id - 2
            before_name = "template_" + str(self.frame_id - 2) + ".png"
            before_path = path + "/" + before_name
            after_name = "template_mask_25_" + str(self.frame_id - 2) + ".png"
            after_path = path + "/template_whole/" + after_name
            before_mask = self.store_mask[idx]
            apply_mask_and_generate_mask_image(image_path=before_path,
                                               mask_tensor=before_mask,
                                               output_image_path=after_path)
            after_name = "template_mask_50_" + str(self.frame_id - 2) + ".png"
            after_path = path + "/template_whole/" + after_name
            before_mask = self.store_mask_50[idx]
            apply_mask_and_generate_mask_image(image_path=before_path,
                                               mask_tensor=before_mask,
                                               output_image_path=after_path)
            after_name = "template_mask_75_" + str(self.frame_id - 2) + ".png"
            after_path = path + "/template_whole/" + after_name
            before_mask = self.store_mask_75[idx]
            apply_mask_and_generate_mask_image(image_path=before_path,
                                               mask_tensor=before_mask,
                                               output_image_path=after_path)
            after_name = "template_mask_90_" + str(self.frame_id - 2) + ".png"
            after_path = path + "/template_whole/" + after_name
            before_mask = self.store_mask_90[idx]
            apply_mask_and_generate_mask_image(image_path=before_path,
                                               mask_tensor=before_mask,
                                               output_image_path=after_path)



        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
       # if self.dz_feat == None:
       #     self.dz_feat = self.network.backbone.patch_embed(self.z_dict2.tensors)
        for i in range(len(self.store_result)):
            box_temp = self.store_result[i].copy()
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)

        seqs_out = seqs_out.unsqueeze(0)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
          #  if self.update_:
          #      template = torch.concat([self.z_dict1.tensors.unsqueeze(1), self.z_dict2.unsqueeze(1)], dim=1)
          #  else:
          #      template = torch.concat([self.z_dict1.tensors.unsqueeze(1), self.z_dict2.tensors.unsqueeze(1)], dim=1)
            out_dict = self.network.forward(
                template=self.z_dict1, search=x_dict.tensors, ce_template_mask=self.box_mask_z,
                seq_input=seqs_out, stage="inference", search_feature=None, mask=self.mask)


        #self.dz_feat = out_dict['dz_feat']
        #self.x_feat = out_dict['x_feat']
        mask = out_dict['mask']

        pred_boxes = (out_dict['seqs'][:, 0:4] + 0.5) / (self.bins - 1) - 0.5

        pred_feat = out_dict['feat']
        pred = pred_feat.permute(1, 0, 2).reshape(-1, self.bins * self.range + 5)

        pred = pred_feat[0:4, :, 0:self.bins * self.range]

        out = pred.softmax(-1).to(pred)
        #mul = torch.range((-1 * self.range * 0.5 + 0.5) + 1 / (self.bins * self.range), (self.range * 0.5 + 0.5) - 1 / (self.bins * self.range), 2 / (self.bins * self.range)).to(pred)
        mul = torch.range((-1 * self.range * 0.5 + 0.5), (self.range * 0.5 + 0.5) - 1 / (self.bins * self.range), 2 / (self.bins * self.range)).to(pred)

        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)
        
        #pred_boxes = ans

        pred_boxes = (ans + pred_boxes) / 2

        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)

        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_new[2] / 2
        pred_new[1] = pred_boxes[1] + pred_new[3] / 2

        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
        new_z = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors   
       # self.template_update(new_z, 0)
        self.template_update_sampling(new_z, "exponential", mask=mask) # exponential

        if len(self.store_result) < self.prenum:
            self.store_result.append(self.state.copy())
        else:
            for i in range(self.prenum):
                if i != self.prenum - 1:
                    self.store_result[i] = self.store_result[i + 1]
                else:
                    self.store_result[i] = self.state.copy()

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        # cx_real = cx + cx_prev
        # cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return ARTrackV2Seq

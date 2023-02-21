# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from cmath import inf
import logging
from turtle import width
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
import matplotlib.font_manager as fm # to create font
from PIL import Image,ImageFont,ImageDraw

from detectron2.modeling import GeneralizedRCNN, META_ARCH_REGISTRY
from detectron2.data import detection_utils as d2utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from glsan.utils import uniformly_crop, self_adaptive_crop, cluster_by_boxes_centers
from .edsr import EDSR
import math

__all__ = ["GlsanNet"]


@META_ARCH_REGISTRY.register()
class GlsanNet(GeneralizedRCNN):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.image_format = cfg.INPUT.FORMAT
        self.crop_method = cfg.GLSAN.CROP
        self.cluster_num = cfg.GLSAN.CLUSTER_NUM
        self.crop_size = cfg.GLSAN.CROP_SIZE
        self.padding_size = cfg.GLSAN.PADDING_SIZE
        self.normalized_ratio = cfg.GLSAN.NORMALIZED_RATIO
        self.sr = cfg.GLSAN.SR
        self.sr_thresh = cfg.GLSAN.SR_THRESH
        self.sr_model = EDSR().to(self.device)
        self.sr_model.load_state_dict(torch.load('./models/visdrone_x2.pt'))
        self.history_transform = nn.Sequential( 
            nn.Conv2d(in_channels=1, out_channels=64,  kernel_size=3, stride=2, padding=1, bias=True ),
        )

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        #print(images.tensor.shape)
        batch_size,height,width = images.tensor.shape[0], images.tensor.shape[2], images.tensor.shape[3]
        history = None
        history_enable = True
        if history_enable:
            history = torch.zeros((batch_size, 1,height,width), device = self.device)
        

        
        if not self.training:
            return self.inference(batched_inputs, history)
            #return self.inference(batched_inputs)
 
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        if history_enable:
            history = self.history_transform(history)
        #history = None
        features = self.backbone(images.tensor, history)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        results, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        
        results = GlsanNet._postprocess(results, batched_inputs, images.image_sizes)
        
        
        if proposal_losses['loss_rpn_loc'].item() == inf:
            
            del proposal_losses['loss_rpn_loc']
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.crop_method == "NoCrop":
            return losses
#######################################增加第二阶段迭代loss###################################
        test_images = []
        offsets = []
        part_imgs = []
        aug_images = []
        augs = d2utils.build_augmentation(self.cfg, self.training)
        part_results = []
        gt_clus_boxes_offset = []
        for i in range(len(batched_inputs)):
            with torch.no_grad():
                #print(results[i]['instances'].pred_boxes.__dict__['tensor'].cpu().numpy())
                if len(results[i]['instances'].pred_classes.cpu().numpy()) == 0:
                    print('result==0')
                    continue
            image = d2utils.read_image(batched_inputs[i]["file_name"], format=self.image_format)
            test_images.append(image)
            if self.crop_method == "UniformlyCrop":
                offsets_per_img, part_imgs_per_img = uniformly_crop(image)
            elif self.crop_method == "SelfAdaptiveCrop":
                with torch.no_grad():
                    offsets_per_img, part_imgs_per_img, historys,gt_clus_boxes, gt_clus_classes = \
                        self_adaptive_crop(results[i]['instances'].pred_boxes.__dict__['tensor'].cpu().numpy(), results[i]['instances'].pred_classes.cpu().numpy(),image, self.cluster_num,
                                        self.crop_size, self.padding_size, self.normalized_ratio, './visualization/dbsan_box/' + batched_inputs[i]["file_name"].split('/')[-1], gt_instances[i])#增加类别的输入，直接返回分割好的图片
            if offsets_per_img == []:
                continue
            offsets.append(offsets_per_img)
            part_imgs.append(part_imgs_per_img)
            

            



            aug_inputs_per_img = []
            history_tensors = []
            gt_instances_new = []
            for img_i in range(len(part_imgs_per_img)):
                image = part_imgs_per_img[img_i]
                image_shape = image.shape[0:2]
                if image_shape[0] == 0 or image_shape[1] == 0:
                    continue
                
                if self.sr:
                    # super-resolution
                    image_size = math.sqrt(image_shape[0] * image_shape[1])
                    if image_size <= self.sr_thresh:
                        sr_input = torch.FloatTensor(image.copy()).to(self.device).permute(2, 0, 1).unsqueeze(0)
                        image = self.sr_model(sr_input)
                        image = image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                aug_input = T.StandardAugInput(image)
                aug_input.apply_augmentations(augs)
                aug_image = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(self.device)
                part_aug_input = dict()
                part_aug_input['image'] = aug_image
                part_aug_input['height'], part_aug_input['width'] = image_shape
                part_aug_input['file_name'] = batched_inputs[i]['file_name']
                part_aug_input['image_id'] = batched_inputs[i]['image_id']
                aug_inputs_per_img.append(part_aug_input)
                part_img_radio = part_aug_input['image'].size(1)/part_aug_input['height']

                gt_clus_boxes_tmp = []
                for g in range(len(gt_clus_boxes[img_i])):
                    gt_tmp = gt_clus_boxes[img_i][g].copy()
                    gt_tmp[0] = max(math.ceil(gt_clus_boxes[img_i][g][0] - offsets_per_img[img_i][1])*part_img_radio,0)
                    gt_tmp[1] = max(math.ceil(gt_clus_boxes[img_i][g][1] - offsets_per_img[img_i][0])*part_img_radio,0)
                    gt_tmp[2] = max(math.ceil(gt_clus_boxes[img_i][g][2] - offsets_per_img[img_i][1])*part_img_radio,0)
                    gt_tmp[3] = max(math.ceil(gt_clus_boxes[img_i][g][3] - offsets_per_img[img_i][0])*part_img_radio,0)
                    gt_clus_boxes_tmp.append(gt_tmp)
                gt_clus_boxes_tmp_tensor = torch.tensor(gt_clus_boxes_tmp,device= self.device)
                gt_part_box = Boxes(gt_clus_boxes_tmp_tensor)
                gt_part_classes = torch.tensor(gt_clus_classes[img_i],device=self.device)
                boxes_dict = {}
                boxes_dict['gt_boxes'] = gt_part_box
                boxes_dict['gt_classes'] = gt_part_classes
                
                instances = Instances((part_aug_input['image'].size(1),part_aug_input['image'].size(2)))
                instances.gt_boxes = gt_part_box
                instances.gt_classes = gt_part_classes
                gt_instances_new.append(instances)

                history_aug_input = T.StandardAugInput(historys[img_i])
                history_aug_input.apply_augmentations(augs)
                history_aug_image = torch.as_tensor(np.ascontiguousarray(history_aug_input.image.transpose(2, 0, 1))).to(self.device)

                
                history_tensors.append(history_aug_image)

            history = history_tensors
            images = self.preprocess_image(aug_inputs_per_img)
            batch_size,height,width = images.tensor.shape[0], images.tensor.shape[2], images.tensor.shape[3]
            history_list = None
            if history != None:
                if torch.is_tensor(history):
                    history_out = history
                else:
                    for his_i in range(len(history)):
                        if history[his_i].shape[1] != height or history[his_i].shape[2] != width:
                            pad = nn.ZeroPad2d(padding = (0, width - history[his_i].shape[2],0, height - history[his_i].shape[1]))
                            test_history = pad(history[his_i])
                            if history_list == None:
                                history_list = pad(history[his_i])
                            else:
                                history_list = torch.cat((history_list, pad(history[his_i])))
                    
                    if history_list == None:
                        history = None 
                    else:
                        history_out = history_list.type(torch.FloatTensor).unsqueeze(1).cuda()
                if history_list == None:
                    history = None 
                else:
                    #history_out_new = torch.ones(history_out.shape[0],1,history_out.shape[2],history_out.shape[3])
                    history = self.history_transform(history_out)
            
            if history_enable is not True:
                history = None

            features = self.backbone(images.tensor, history)

            
            if self.proposal_generator:
                None
                proposals, part_proposal_losses = self.proposal_generator(images, features, gt_instances_new)
            else:
                assert "proposals" in aug_inputs_per_img[0]
                proposals = [x["proposals"].to(self.device) for x in aug_inputs_per_img]

            results, part_detector_losses = self.roi_heads(images, features, proposals, gt_instances_new)
            if 'loss_rpn_loc' in part_proposal_losses.keys():
                if part_proposal_losses['loss_rpn_loc'] != inf:
                    part_proposal_losses.update({str(i)+'_loss_rpn_loc':part_proposal_losses.pop('loss_rpn_loc')})
            if 'loss_rpn_cls' in part_proposal_losses.keys():
                part_proposal_losses.update({str(i)+'_loss_rpn_cls':part_proposal_losses.pop('loss_rpn_cls')})
            if 'loss_cls' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_cls':part_detector_losses.pop('loss_cls')})
            if 'loss_box_reg' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_box_reg':part_detector_losses.pop('loss_box_reg')})
            if 'loss_cls_stage0' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_cls_stage0':part_detector_losses.pop('loss_cls_stage0')})
            if 'loss_box_reg_stage0' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_box_reg_stage0':part_detector_losses.pop('loss_box_reg_stage0')})
            if 'loss_cls_stage1' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_cls_stage1':part_detector_losses.pop('loss_cls_stage1')})
            if 'loss_box_reg_stage1' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_box_reg_stage1':part_detector_losses.pop('loss_box_reg_stage1')})
            if 'loss_cls_stage2' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_cls_stage2':part_detector_losses.pop('loss_cls_stage2')})
            if 'loss_box_reg_stage2' in part_detector_losses.keys():
                part_detector_losses.update({str(i)+'_loss_box_reg_stage2':part_detector_losses.pop('loss_box_reg_stage2')})
            losses.update(part_proposal_losses)
            losses.update(part_detector_losses)






        torch.cuda.empty_cache()
        return losses

    def inference(self,batched_inputs, history=None, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        assert self.crop_method in ["NoCrop", "UniformlyCrop", "SelfAdaptiveCrop"], "crop method not in given range!"
        results = self.batched_inference(batched_inputs, detected_instances, do_postprocess, history)
        for r_i in range(len(results)):
            image = d2utils.read_image(batched_inputs[r_i]["file_name"], format=self.image_format)
            # self.visualize_boxes(results[r_i], image.copy(),
            #                      './visualization/baseline/' + batched_inputs[r_i]["file_name"].split('/')[-1],
            #                      show_score=False, show_class=True)
        if self.crop_method == "NoCrop":
            return results
        images = []
        offsets = []
        part_imgs = []
        aug_images = []
        augs = d2utils.build_augmentation(self.cfg, self.training)
        part_results = []
        for i in range(len(batched_inputs)):
            with torch.no_grad():
                #print(results[i]['instances'].pred_boxes.__dict__['tensor'].cpu().numpy())
                if len(results[i]['instances'].pred_classes.cpu().numpy()) == 0:
                    print('result==0')
                    continue
            image = d2utils.read_image(batched_inputs[i]["file_name"], format=self.image_format)
            images.append(image)
            if self.crop_method == "UniformlyCrop":
                offsets_per_img, part_imgs_per_img = uniformly_crop(image)
            elif self.crop_method == "SelfAdaptiveCrop":
                offsets_per_img, part_imgs_per_img, historys, _, _ = \
                    self_adaptive_crop(results[i]['instances'].pred_boxes.tensor.cpu().numpy(), results[i]['instances'].pred_classes.cpu().numpy(),image, self.cluster_num,
                                       self.crop_size, self.padding_size, self.normalized_ratio, './visualization/dbsan_box/' + batched_inputs[r_i]["file_name"].split('/')[-1])#增加类别的输入，直接返回分割好的图片
            offsets.append(offsets_per_img)
            part_imgs.append(part_imgs_per_img)

            aug_inputs_per_img = []
            history_tensors = []
            for img_i in range(len(part_imgs_per_img)):
                image = part_imgs_per_img[img_i]
                image_shape = image.shape[0:2]
                if self.sr:
                    # super-resolution
                    image_size = math.sqrt(image_shape[0] * image_shape[1])
                    if image_size <= self.sr_thresh:
                        sr_input = torch.FloatTensor(image.copy()).to(self.device).permute(2, 0, 1).unsqueeze(0)
                        image = self.sr_model(sr_input)
                        image = image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                aug_input = T.StandardAugInput(image)
                if aug_input.image.shape[0] == 0 or aug_input.image.shape[1] == 0:
                    print('skip!!!!')
                    continue
                aug_input.apply_augmentations(augs)
                aug_image = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(self.device)
                part_aug_input = dict()
                part_aug_input['image'] = aug_image
                part_aug_input['height'], part_aug_input['width'] = image_shape
                part_aug_input['file_name'] = batched_inputs[i]['file_name']
                part_aug_input['image_id'] = batched_inputs[i]['image_id']
                aug_inputs_per_img.append(part_aug_input)

                history_aug_input = T.StandardAugInput(historys[img_i])
                history_aug_input.apply_augmentations(augs)
                history_aug_image = torch.as_tensor(np.ascontiguousarray(history_aug_input.image.transpose(2, 0, 1))).to(self.device)

                
                history_tensors.append(history_aug_image)
            
            # history_array = np.array(historys) 
            # history_tensor = torch.from_numpy(history_array)
            # history_tensor = history_tensor.type(torch.FloatTensor).cuda()
            # history_tensor = None


            part_result = self.batched_inference(aug_inputs_per_img, detected_instances, do_postprocess, history_tensors)
            part_results.append(part_result)
            aug_images.append(aug_inputs_per_img)
        merged_results = self.merge_results(results, part_results, offsets, merge_mode='merge')
        # for r_i in range(len(merged_results)):
        #     self.visualize_boxes(merged_results[r_i], images[r_i].copy(),
        #                          './visualization/ori/' + batched_inputs[r_i]["file_name"].split('/')[-1],
        #                          show_score=False, show_class=True)
        return merged_results

    def batched_inference(self, batched_inputs, detected_instances=None, do_postprocess=True, history = None):
        if len(batched_inputs) == 0:
            return []
        images = self.preprocess_image(batched_inputs)
        batch_size,height,width = images.tensor.shape[0], images.tensor.shape[2], images.tensor.shape[3]
        history_list = None
        if history != None:
            if torch.is_tensor(history):
                history_out = history
            else:
                for i in range(len(history)):
                    if history[i].shape[1] != height or history[i].shape[2] != width:
                        pad = nn.ZeroPad2d(padding = (0, width - history[i].shape[2],0, height - history[i].shape[1]))
                        test_history = pad(history[i])
                        if history_list == None:
                            history_list = pad(history[i])
                        else:
                            history_list = torch.cat((history_list, pad(history[i])))
                
                if history_list == None:
                    history = None 
                else:
                    history_out = history_list.type(torch.FloatTensor).unsqueeze(1).cuda()
            if history_list == None:
                history = None 
            else:
                history = self.history_transform(history_out)
        features = self.backbone(images.tensor, history)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GlsanNet._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def merge_results(self, results, part_results, offsets, merge_mode='merge'):
        result_per_image = [
            self.merge_results_single_image(result, part_result, offset, merge_mode)
            for result, part_result, offset in zip(results, part_results, offsets)
        ]
        return result_per_image

    def merge_results_single_image(self, result, part_result, offset, merge_mode='merge'):
        assert merge_mode in ['global', 'local', 'merge'], 'merge mode must in [\'global\', \'local\', \'merge\']!'
        if len(part_result) == 0:
            return result
        result = result['instances']
        merged_result = Instances(result.image_size)
        if merge_mode == 'global':
            return {'instances': result}
        elif merge_mode == 'local':
            merged_boxes = []
            merged_scores = []
            merged_pred_classes = []
        else:
            merged_boxes = [result.pred_boxes.tensor]
            merged_scores = [result.scores]
            merged_pred_classes = [result.pred_classes]

        for i in range(len(part_result)):
            part_instance = part_result[i]['instances']
            part_boxes = part_instance.pred_boxes.tensor
            part_offset = torch.tensor(offset[i]).to(self.device).flip(0).repeat(part_boxes.shape[0], 2)

            merged_boxes.append(part_boxes + part_offset)
            merged_scores.append(part_instance.scores)
            merged_pred_classes.append(part_instance.pred_classes)

        merged_boxes = torch.cat(merged_boxes, dim=0)
        merged_scores = torch.cat(merged_scores, dim=0)
        merged_pred_classes = torch.cat(merged_pred_classes, dim=0)

        # Apply per-class NMS
        keep = batched_nms(merged_boxes, merged_scores, merged_pred_classes, self.test_nms_thresh)
        if self.test_topk_per_image >= 0:
            keep = keep[:self.test_topk_per_image]
        boxes, scores, pred_classes = merged_boxes[keep], merged_scores[keep], merged_pred_classes[keep]

        merged_result.pred_boxes = Boxes(boxes)
        merged_result.scores = scores
        merged_result.pred_classes = pred_classes
        return {'instances': merged_result} 

    def visualize_boxes(self, result, image, file_name, show_score=False, show_class=True):
        abbr_classes = ['PT','BT','CT','WT','HB','UT','YP','PG','SH','SP','GR','SR','WE','AB','IN']
        img = Image.fromarray(image[...,::-1])
        draw = ImageDraw.Draw(img)
        pred_boxes = result['instances'].pred_boxes.tensor.cpu().numpy().astype(np.int32)
        scores = result['instances'].scores.cpu().numpy()
        pred_classes = result['instances'].pred_classes.cpu().numpy()
        meta = MetadataCatalog.get('visdrone_train')
        #font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='LiberationSans-Regular.ttf')), 20)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 40)
        for box_i, pred_box in enumerate(pred_boxes):
            if scores[box_i] < 0.3: continue
            color = tuple(meta.thing_colors[pred_classes[box_i]])
            draw.rectangle([pred_box[0], pred_box[1], pred_box[2], pred_box[3]], outline=color,width=5)
            if show_score:
                score = scores[box_i]
                draw.text((pred_box[2], pred_box[1]),
                          str(np.around(score, decimals=2)), font=font, fill=color)
            if show_class:
                draw.rectangle([pred_box[0]+5, pred_box[1]+5, pred_box[0]+60, pred_box[1]+38], fill='#Ffffff',outline='#Ffffff')
                pred_class = abbr_classes[pred_classes[box_i]]
                draw.text((pred_box[0]+5, pred_box[1]), pred_class, font=font, fill='#000000')
        img.save(file_name)
    
    

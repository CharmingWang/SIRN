import numpy as np
import time
import difflib
from sklearn.cluster import KMeans
import scipy
import math
from PIL import Image,ImageFont,ImageDraw
import matplotlib.font_manager as fm # to create font
__all__ = ['uniformly_crop', 'self_adaptive_crop', 'cluster_by_boxes_centers']

class Dbscan():
    def __init__(self,eps,MinPts):
        self.eps = eps
        self.MinPts = MinPts


    ##########################生成表格记录点与点之间的距离######################################
    # 函数计算数据文件中每个点之间的距离，并返回Mydist。用来判断这个点是属于核心点、边界点还是噪音点
    def distance(self,data,img_size):
        h = img_size[0]
        w = img_size[1]
        data = np.array(data)
        m,n= np.shape(data)
        Mydist = np.mat(np.zeros((m, m)))
        for a in range(m):
            for z in range(a, m):
                # 计算a和z之间的欧式距离
                tmp = 0
                for k in range(n):
                    tmp += (data[a, k] - data[z, k]) * (data[a, k] - data[z, k])
                #tmp_new = tmp/(h^2+w^2)
                Mydist[a, z] = np.sqrt(tmp)
                Mydist[z, a] = Mydist[a, z]
        return Mydist

    ##########################################################################
    # 参数 distance: 存储了所有数据样本点之间距离的向量
    # 参数 Myradius: 定义的样本半径
    # return: X:存储符合半径范围内的样本下标的向量
    def find_in_radius(self,distance, Myradius):
        X = []
        n = np.shape(distance)[1]
        for j in range(n):
            if distance[0, j] <= Myradius:
                X.append(j)
        return X

    ###########################################################################################
    # 在DBSCAN算法中，从核心对象出发，找到与该核心对象密度可达的所有样本形成“簇”。DBSCAN算法的流程为：
    # 1.根据给定的eps和MinPts确定所有的核心对象，eps是定义密度时的邻域半径，MinPts是定义核心点时的阈值。在 DBSCAN 算法中将数据点分为3类。
    # （1）核心点：在其半径eps内含有超过MinPts数目的点为核心点；（2）边界点：在其半径eps内含有点的数量小于MinPts，但是落在核心点的邻域内，则为边界点。（3）噪音点：一个对点既不是核心点也不是边界点，则为噪音点。
    # 2.对每一个核心对象，选择一个没有处理过的核心对象，找到由其密度可达的的样本生成聚类“簇”
    # 3.重复以上过程，直到所有的点都被处理。
    def dbscan(self,data, eps, MinPts, img_size):
        m = np.shape(data)[0]  # m为样本个数
        # 区分核心点1，边界点0和噪音点-1
        types = np.mat(np.zeros((1, m)))  # 初始化，每个点的类型
        sub_class = np.mat(np.zeros((1, m)))  # 初始化，每个点所属类别
        # 判断该点是否处理过，0表示未处理过
        did = np.mat(np.zeros((m, 1)))
        # 计算每个数据点之间的距离
        dis = self.distance(data,img_size)
        # 用于标记类别
        number = 1
        # 对每一个点进行处理
        for i in range(m):
            # 找到未处理的点
            #print('dealing.......')
            if did[i, 0] == 0:
                # 找到第i个点到其他所有点的距离
                D = dis[i,]
                # 找到半径r内的所有点
                X = self.find_in_radius(D, eps)  # X[]只存了下标
                # 区分点的类型
                # 边界点
                if len(X) > 1 and len(X) < MinPts + 1:
                    types[0, i] = 0
                    sub_class[0, i] = 0  # 类别为0，表示尚未处理过的
                # 噪音点
                if len(X) == 1:
                    types[0, i] = -1
                    sub_class[0, i] = -1
                    did[i, 0] = 1
                # 核心点
                if len(X) >= MinPts + 1:
                    types[0, i] = 1
                    for x in X:
                        sub_class[0, x] = number
                    # 判断核心点是否密度可达
                    while len(X) > 0:
                        # 循环去找eps范围内的点（每个点下标都为X[0],因为处理完就会删除，相当于队列），标注相同的所属类别
                        # 延伸出去，找到的点X[0]半径范围内的点（下标存在ind_1中）：
                        # 全部标为相同类别
                        # 若之前未被处理，则标为已经处理过；再加入X列表中，参与循环延伸
                        did[X[0], 0] = 1
                        D = dis[X[0],]
                        tmp = X[0]
                        del X[0]
                        ind_1 = self.find_in_radius(D, eps)

                        if len(ind_1) > 1:  # 处理非噪音点
                            for x1 in ind_1:
                                sub_class[0, x1] = number
                            for j in range(len(ind_1)):
                                if did[ind_1[j], 0] == 0:
                                    did[ind_1[j], 0] = 1
                                    X.append(ind_1[j])
                                    sub_class[0, ind_1[j]] = number
                    number += 1
        # 最后处理所有未分类的点为噪音点
        X_2 = ((sub_class == 0).nonzero())[1]
        for x in X_2:
            sub_class[0, x] = -1
            types[0, x] = -1
        return types, sub_class

    ##########################可以不保存###############################
    # 1.保存每个点的类型，其中，核心点为1，边界点为0，噪音点为-1，在主函数中生成types文件
    # 2.保存每个点所属类别，在主函数中生成sub_class文件
    def save_data(self,file_name, source):
        f = open(file_name, "w")
        n = np.shape(source)[1]
        tmp = []
        for i in range(n):
            tmp.append(str(source[0, i]))
        f.write("\n".join(tmp))
        f.close()

    ###################################画图的类别数不定，看聚类结果###########################
    # 根据sub_class文件中生成的数据1.0,2.0,3.0,4.0,5.0，共计五个类别，对每个点的所属类别进行绘图，其中每种类别分别用不同的颜色来表示，便于图像观察与分析
    def draw(self,point_data, subclass):
        Myfig = plt.figure()
        axes = Myfig.add_subplot(111)
        length = len(point_data)
        for j in range(length):
            if subclass[0, j] == 1:
                axes.scatter(point_data[j, 0], point_data[j, 1], color='c', alpha=0.4)
            if subclass[0, j] == 2:
                axes.scatter(point_data[j, 0], point_data[j, 1], color='g', alpha=0.4)
            if subclass[0, j] == 3:
                axes.scatter(point_data[j, 0], point_data[j, 1], color='m', alpha=0.4)
            if subclass[0, j] == 4:
                axes.scatter(point_data[j, 0], point_data[j, 1], color='y', alpha=0.4)
            if subclass[0, j] == 5:
                axes.scatter(point_data[j, 0], point_data[j, 1], color='r', alpha=0.4)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('dbscan-JYZ')
        plt.show()

    def begin_cluster(self,sample_data,img_size):
        #print('Begain to culster!!!!!!!!')
        #print('MinPts = {}'.format(self.MinPts))
        #print('eps = {}'.format(self.eps))

        types, sub_class = self.dbscan(sample_data,self.eps,self.MinPts,img_size)
        #self.draw(sample_data, sub_class)
        return types,sub_class



def uniformly_crop(img_ori):
    height_ori = img_ori.shape[0]
    width_ori = img_ori.shape[1]
    offsets = []
    imgs = []
    offsets.append([0, 0])
    imgs.append(img_ori[0:height_ori // 2, 0:width_ori // 2, :])

    offsets.append([0, width_ori // 2])
    imgs.append(img_ori[0:height_ori // 2, width_ori // 2:, :])

    offsets.append([height_ori // 2, 0])
    imgs.append(img_ori[height_ori // 2:, 0:width_ori // 2, :])

    offsets.append([height_ori // 2, width_ori // 2])
    imgs.append(img_ori[height_ori // 2:, width_ori // 2:, :])

    return offsets, imgs


def self_adaptive_crop(boxes, classes, img_ori, cluster_num=4, crop_size=300, padding_size=50, normalized_ratio=2, file_name='1', gt_instances = None):
    height_ori = img_ori.shape[0]
    width_ori = img_ori.shape[1]
    gt_boxes_ori = None
    gt_classes = None
    img_size = [height_ori,width_ori]
    center_map = np.zeros((height_ori, width_ori))
    historys = []
    if gt_instances is not None:
        gt_boxes = gt_instances.gt_boxes.__dict__['tensor'].cpu().numpy()
        gt_classes = gt_instances.gt_classes.cpu().numpy()
        y_change_ratio = height_ori / gt_instances.image_size[0]
        x_change_ratio = width_ori / gt_instances.image_size[1]
        gt_boxes_ori = []
        for gt_box in gt_boxes:
            gt_boxes_ori.append([gt_box[0] * x_change_ratio,gt_box[1] * y_change_ratio,gt_box[2] * x_change_ratio, gt_box[3] * y_change_ratio])
        #visualize_dbsan_boxes(gt_boxes_ori,img_ori,file_name.split('.jpg')[0] + '_gtbox.jpg')


    # centers, ranges = cluster_by_boxes_centers(cluster_num, center_map, boxes,
    #                                            crop_size, padding_size, normalized_ratio)
    centers, ranges, history, gt_clus_boxes, gt_clus_classes = cluster_by_dbscan(img_size, center_map, boxes, classes,crop_size, padding_size, normalized_ratio, gt_boxes_ori, gt_classes)
    

    imgs = []
    offsets = []
    all_boxes = []
    for i, center in enumerate(centers):
        range = ranges[i]
        # part_x1, part_y1, part_x2, part_y2 = clamp_range(center, range, height_ori, width_ori)
        center_x = center[1]
        center_y = center[0]
        crop_width = range[1]
        crop_height = range[0]
        part_x1 = int(center_x - crop_width // 2)
        part_x2 = int(center_x + crop_width // 2)
        part_y1 = int(center_y - crop_height // 2)
        part_y2 = int(center_y + crop_height // 2)
        #part_x1, part_y1, part_x2, part_y2 = transfrom_offsets(center, range, height_ori, width_ori)
        part_img = img_ori[part_y1:part_y2, part_x1:part_x2, :]
        part_history = history[0,0,part_y1:part_y2, part_x1:part_x2]
        #part_history = scipy.ndimage.filters.gaussian_filter(part_history, 8, mode='constant')
        part_history_list = part_history[:,:,np.newaxis]
        all_boxes.append([part_x1, part_y1, part_x2, part_y2])
        offsets.append([part_y1, part_x1])
        imgs.append(part_img)
        historys.append(part_history_list)
    pre_boxes = boxes.tolist()
    #visualize_dbsan_boxes(pre_boxes,img_ori,file_name.split('.jpg')[0] + '_prebox.jpg')
    visualize_dbsan_boxes(all_boxes,img_ori,file_name)
    visualize_dbsan_boxes_all(gt_clus_boxes, all_boxes,img_ori, file_name.split('.jpg')[0] + '_allbox.jpg')


    return offsets, imgs, historys, gt_clus_boxes, gt_clus_classes

def visualize_dbsan_boxes(boxes, image, file_name):
        img = Image.fromarray(image[...,::-1])
        draw = ImageDraw.Draw(img)
        
        #font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='LiberationSans-Regular.ttf')), 20)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 20)
        for box_i, pred_box in enumerate(boxes):
            
            color = (255,255,255)
            draw.rectangle([pred_box[0], pred_box[1], pred_box[2], pred_box[3]], outline=color,width=10)
            
        img.save(file_name)

def visualize_dbsan_boxes_all(gt_clus_boxes, all_boxes,image, file_name):
        img = Image.fromarray(image[...,::-1])
        draw = ImageDraw.Draw(img)
        for box_i, pred_box in enumerate(all_boxes):
            
            color = (255,255,255)
            draw.rectangle([pred_box[0], pred_box[1], pred_box[2], pred_box[3]], outline=color,width=10)

        #font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='LiberationSans-Regular.ttf')), 20)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 20)
        for boxes in gt_clus_boxes:
            for pred_box in boxes:
                color = (255,255,0)
                draw.rectangle([pred_box[0], pred_box[1], pred_box[2], pred_box[3]], outline=color,width=5)



        
            
        img.save(file_name)




def clamp_range(center, range, height_ori, width_ori):
    # [x,y,x,y]
    r = [max(0, center[1] - range[1] // 2),
         max(0, center[0] - range[0] // 2),
         min(width_ori, center[1] + range[1] // 2),
         min(height_ori, center[0] + range[0] // 2)]
    return r

def get_index(lst=None, item=''):   #用来获得一个列表中，某个相同元素的所有索引，如[1，2，1，1，2]，item设为2，则返回[1，4]
    return [index for (index, value) in enumerate(lst) if value == item]


def transfrom_offsets(center, range, height_ori, width_ori):
    center_x = center[1]
    center_y = center[0]
    crop_width = range[1]
    crop_height = range[0]
    part_x1 = int(center_x - crop_width // 2)
    part_x2 = int(center_x + crop_width // 2)
    part_y1 = int(center_y - crop_height // 2)
    part_y2 = int(center_y + crop_height // 2)
    if part_x1 < 0 and part_x2 > width_ori:
        center_x = int(width_ori // 2)
        part_x1 = 0
        part_x2 = width_ori
    elif part_x1 < 0:
        offset_x = 0 - part_x1
        center_x += offset_x
        part_x1 += offset_x
        part_x2 += offset_x
        if part_x2 > width_ori:
            center_x += (width_ori - part_x2) / 2
            part_x2 = width_ori
    elif part_x2 > width_ori:
        offset_x = width_ori - part_x2
        center_x += offset_x
        part_x1 += offset_x
        part_x2 += offset_x
        if part_x1 < 0:
            center_x += (0 - part_x1) / 2
            part_x1 = 0
    if part_y1 < 0 and part_y2 > height_ori:
        center_y = int(height_ori // 2)
        part_y1 = 0
        part_y2 = height_ori
    elif part_y1 < 0:
        offset_y = 0 - part_y1
        center_y += offset_y
        part_y1 += offset_y
        part_y2 += offset_y
        if part_y2 > height_ori:
            center_y += (height_ori - part_y2) / 2
            part_y2 = height_ori
    elif part_y2 > height_ori:
        offset_y = height_ori - part_y2
        center_y += offset_y
        part_y1 += offset_y
        part_y2 += offset_y
        if part_y1 < 0:
            center_y += (0 - part_y1) / 2
            part_y1 = 0
    return part_x1, part_y1, part_x2, part_y2

def cluster_by_dbscan(img_size, center_map, boxes, classes,crop_size=300, padding_size=50, normalized_ratio=2, gt_boxes_ori = None, gt_classes = None):
    start = time.time()
    center2boxes = {}
    all_boxes_center_xy = []
    ranges = []#以列表的形式存储每个聚类框的实际宽和高
    centers = []#以列表的形式存储每个聚类框的中心点坐标
    history_boxes = []#存储每个聚类区域上一级检测结果
    gt_clus_boxes = []#存储每个聚类区域的真实标签框
    gt_clus_classes = []#存储每个聚类区域的真实标签类别
    for p in boxes:
        x = int(p[0] + p[2]) // 2
        y = int(p[1] + p[3]) // 2
        if [y, x] not in all_boxes_center_xy:
            all_boxes_center_xy.append([y, x])
    h = img_size[0]
    w = img_size[1]
    history = np.zeros([1,1,h,w])
    


    radius = math.ceil(math.sqrt((h*h) + (w*w))*0.2)
    #radius = math.ceil(max(h,w)//6)
    dbscan_real = Dbscan(eps=radius, MinPts=2)
    type_done, sub_class_done = dbscan_real.begin_cluster(all_boxes_center_xy,img_size)

    sub_class_list = sub_class_done.tolist()
    sub_class_list = sub_class_list[0]
    sub_class_set = set(sub_class_list)
    scence_list = [[10,1,6,4,14],[1,12,4,10,14],[13,4,5,14],[13,11,4,6,5,2,10,14],[7,3,5,14],[5,6,4,1,14],[7,8,14],[0,5,6,4,14],[7,5,6,14]]

    area = []
    rough_disperse = []

    
    # for k in sub_class_set:  # k的可能值：2，1，0，-1等等...
    #     index = get_index(sub_class_list, k)  # 例如k = 1,就把sub_class_list中得所有k = 1的索引给取出
    #     if k == -1:
    #         for l in index:
    #             centers.append(all_boxes_center_xy[l])
    #             ranges.append([boxes[l][3]-boxes[l][1], boxes[l][2]-boxes[l][0]])
    #     else:
    #         for j in index:
    #             if classes[j] == 14:
    #                 index.remove(j)
    #         tmp = boxes[index, :]
    #         #tmp = tmp[0]
    #         x0_min = min(tmp[:, 0])
    #         y0_min = min(tmp[:, 1])
    #         x1_max = max(tmp[:, 2])
    #         y1_max = max(tmp[:, 3])
    #         ranges.append([y1_max-y0_min, x1_max-x0_min])
    #         centers.append([(y1_max+y0_min)//2, (x1_max+x0_min)//2])
    # ranges = np.asarray(ranges).astype(np.int32)
    # centers = np.asarray(centers).astype(np.int32)
#######################增加场景先验修正聚类框#########################################
    for k in sub_class_set:  # k的可能值：2，1，0，-1等等...
        index = get_index(sub_class_list, k)  # 例如k = 1,就把sub_class_list中得所有k = 1的索引给取出
        if k == -1:
            for l in index:                               
                x_min = int(boxes[l][0])
                y_min = int(boxes[l][1])
                x_max = int(boxes[l][2])
                y_max = int(boxes[l][3])
                if x_max-x_min<300 and y_max-y_min<300:
                    continue
                if gt_boxes_ori is not None:
                    tmp_boxes = []
                    tmp_classes = []
                    for n in range(len(gt_boxes_ori)):
                        if cal_iou_xyxy(gt_boxes_ori[n],[x_min,y_min,x_max,y_max]):
                            gtboxes_tmp = gt_boxes_ori[n]
                            gtboxes_tmp[0] = max(gt_boxes_ori[n][0],x_min)
                            gtboxes_tmp[1] = max(gt_boxes_ori[n][1],y_min)
                            gtboxes_tmp[2] = min(gt_boxes_ori[n][2],x_max)
                            gtboxes_tmp[3] = min(gt_boxes_ori[n][3],y_max)
                            tmp_boxes.append(gtboxes_tmp)
                            tmp_classes.append(gt_classes[n])
                    if tmp_boxes == []:
                        continue

                    gt_clus_boxes.append(tmp_boxes)
                    gt_clus_classes.append(tmp_classes)
                centers.append(all_boxes_center_xy[l])
                ranges.append([boxes[l][3]-boxes[l][1], boxes[l][2]-boxes[l][0]])
                history_boxes.append(boxes[index,:])
                history[0,0,y_min:y_max, x_min:x_max] +=1
        else:
            num = 9
            diff_a_b = ['' for i in range(num)]
            list_set_classes = list(set(classes[index]))
            list_classes = classes[index].tolist()
            for i in range(len(scence_list)):
                if list_set_classes == 1: 
                    break
                if set(classes[index]) <= set(scence_list[i]):
                    break
                else:
                    diff = difflib.SequenceMatcher(None, list_set_classes, scence_list[i])#计算该区域框目标与各个场景之间的相似度
                    diff_a_b[i] = diff.ratio()
            if set(classes[index]) <= set(scence_list[i]):
                    None
            else:
                scence_index = diff_a_b.index(max(diff_a_b))#计算最大相似场景的索引
                if 11 in list_set_classes:
                    scence_index = 3
                diff_scence = list(set(classes[index]).difference(scence_list[scence_index]))#计算最大相似场景的索引
                
                for m in range(len(diff_scence)):#从index中去掉场景中没有的目标索引
                    diff_scence_value = int(diff_scence[m])
                    for n in index:
                        if classes[n] == diff_scence_value:
                            index.remove(n)

            for j in index:
                if classes[j] == 14:
                    index.remove(j)
            tmp = boxes[index, :]
            #tmp = tmp[0]
            if tmp.shape[0] == 0:
                None
            else:
                x0_min = min(tmp[:, 0])
                y0_min = min(tmp[:, 1])
                x1_max = max(tmp[:, 2])
                y1_max = max(tmp[:, 3])
                if x1_max-x0_min<300 and y1_max-y0_min<300:
                    continue
                if gt_boxes_ori is not None:
                    tmp_boxes = []
                    tmp_classes = []
                    for n in range(len(gt_boxes_ori)):
                        if cal_iou_xyxy(gt_boxes_ori[n],[x0_min,y0_min,x1_max,y1_max]):
                            gtboxes_tmp = gt_boxes_ori[n]
                            gtboxes_tmp[0] = max(gt_boxes_ori[n][0],x0_min)
                            gtboxes_tmp[1] = max(gt_boxes_ori[n][1],y0_min)
                            gtboxes_tmp[2] = min(gt_boxes_ori[n][2],x1_max)
                            gtboxes_tmp[3] = min(gt_boxes_ori[n][3],y1_max)
                            tmp_boxes.append(gtboxes_tmp)
                            tmp_classes.append(gt_classes[n])
                    
                    if tmp_boxes == []:
                        continue
                    gt_clus_boxes.append(tmp_boxes)
                    gt_clus_classes.append(tmp_classes)


                #sigma = {0:4.1,1:3.2,2:4.1,3:2,4:1,5:1,6:3.5,7:1,8:2.5,9:3.1,10:4.5,11:7.8,12:6.1,13:1,14:10}
                sigma = {0:1.73278,1:0.77906,2:1.24715,3:0.68874,4:0.16366,5:0.14243,6:0.99201,7:0.21235,8:0.64222,9:0.96933,10:1.58308,11:7.33614,12:2.06926,13:0.30910,14:10.47660}
                for m in index:
                    x_min = int(boxes[m][0])
                    y_min = int(boxes[m][1])
                    x_max = int(boxes[m][2])
                    y_max = int(boxes[m][3])
                    history[0,0,y_min:y_max, x_min:x_max] +=1
                    history[0,0,y_min:y_max, x_min:x_max] = scipy.ndimage.filters.gaussian_filter(history[0,0,y_min:y_max, x_min:x_max], sigma.get(classes[m]), mode='constant')

                



                history_boxes.append(tmp)
                ranges.append([y1_max-y0_min, x1_max-x0_min])
                centers.append([(y1_max+y0_min)//2, (x1_max+x0_min)//2])
    ranges = np.asarray(ranges).astype(np.int32)
    centers = np.asarray(centers).astype(np.int32)
    #history_boxes = np.asarray(history_boxes).astype(np.int32)
#######################增加场景先验修正聚类框#########################################
 



    return centers, ranges, history, gt_clus_boxes, gt_clus_classes


    # 情况3:如果区域内都是离散点,则区域以整张原图为区域


def cal_iou_xyxy(box1,box2):
    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    a1, b1, a2, b2 = box2[0], box2[1], box2[2], box2[3]
    #计算两个框的面积
    # ax = max(x1, a1) # 相交区域左上角横坐标
    # ay = max(y1, b1) # 相交区域左上角纵坐标
    # bx = min(x2, a2) # 相交区域右下角横坐标
    # by = min(y2, b2) # 相交区域右下角纵坐标
	
    # area_N = (x2 - x1) * (y2 - y1)
    # area_M = (a2 - a1) * (b2 - b1)
	
    # w = bx - ax
    # h = by - ay
    # if w<=0 or h<=0:
    #     return 0 # 不相交返回0	
    # area_X = w * h
    center_x = (x1 + x2)/2
    center_y = (y1 + y2)/2
    if center_x <= a2 and center_x>=a1 and center_y<= b2 and center_y>= b1:
        return True
    else:
        return False


    #return area_X / (area_N + area_M - area_X)






def cluster_by_boxes_centers(cluster_num, center_map, boxes,
                             crop_size=300, padding_size=50, normalized_ratio=2, weight_with_area=False):
    start = time.time()
    center2boxes = {}
    X = []
    weighted_X = []
    for p in boxes:
        x = int(p[0] + p[2]) // 2
        y = int(p[1] + p[3]) // 2
        if [y, x] not in X:
            X.append([y, x])
        weight = int((p[2] - p[0]) * (p[3] - p[1])) // 400 + 1
        if weight_with_area:
            for w in range(weight):
                weighted_X.append([y, x])
        center2boxes[(y, x)] = p
    if len(X) == 0:
        return [], []
    ranges = [[] for i in range(cluster_num)]
    centers = [[] for i in range(cluster_num)]

    if len(X) < cluster_num:
        for i in range(cluster_num):
            if i < len(X):
                centers[i] = X[i]
            else:
                centers[i] = X[0]

            if crop_size <= 0:
                ranges[i] = [300, 300]
            else:
                ranges[i] = [crop_size, crop_size]
        return np.asarray(centers), np.asarray(ranges)
    if weight_with_area:
        X = np.asarray(weighted_X)
    else:
        X = np.asarray(X)
    kmeans = KMeans(n_clusters=cluster_num)
    classes = kmeans.fit(X)
    end = time.time()
    cost_time = end - start

    lbs = classes.labels_
    for i in range(cluster_num):
        inds = np.where(lbs == i)
        tmp_h = X[inds[0]][:, 0]
        tmp_w = X[inds[0]][:, 1]
        assert len(tmp_h) > 0, "X len: {},inds len: {}".format(len(X), len(inds[0]))
        list_h = []
        list_w = []
        for j, h in enumerate(tmp_h):
            w = tmp_w[j]
            box = center2boxes[(h, w)]
            list_w.append(box[0])
            list_w.append(box[2])
            list_h.append(box[1])
            list_h.append(box[3])
        min_h = min(list_h)
        max_h = max(list_h)
        min_w = min(list_w)
        max_w = max(list_w)
        max_height = max_h - min_h + padding_size
        max_width = max_w - min_w + padding_size
        # ranges[i].append(max([max_height, max_width // normalized_ratio]))
        # ranges[i].append(max([max_height // normalized_ratio, max_width]))
        ranges[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
        ranges[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
        centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
    ranges = np.asarray(ranges).astype(np.int32)
    centers = np.asarray(centers).astype(np.int32)

    return centers, ranges


def cluster_by_boxes_scatters(cluster_num, score_map, boxes, crop_size=300, padding_size=50, normalized_ratio=2):
    start = time.time()
    scatters = []
    boxes = boxes.astype(np.int32)
    for p in boxes:
        xs = range(p[0], p[2], 10)
        ys = range(p[1], p[3], 10)
        for x in xs:
            for y in ys:
                scatters.append([y, x])
    if len(scatters) == 0:
        return [], []

    scatters = np.asarray(scatters)
    ranges = [[] for i in range(cluster_num)]
    centers = [[] for i in range(cluster_num)]

    if len(scatters) < cluster_num:
        for i in range(cluster_num):
            if i < len(scatters):
                centers[i] = scatters[i]
                ranges[i] = [crop_size, crop_size]
            else:
                centers[i] = scatters[0]
                ranges[i] = [crop_size, crop_size]
        return np.asarray(centers), np.asarray(ranges)

    kmeans = KMeans(n_clusters=cluster_num)
    classes = kmeans.fit(scatters)
    end = time.time()
    cost_time = end - start

    lbs = classes.labels_
    for i in range(cluster_num):
        inds = np.where(lbs == i)
        list_h = scatters[inds[0]][:, 0]
        list_w = scatters[inds[0]][:, 1]
        min_h = min(list_h)
        max_h = max(list_h)
        min_w = min(list_w)
        max_w = max(list_w)
        max_height = max_h - min_h + padding_size
        max_width = max_w - min_w + padding_size
        # ranges[i].append(max([max_height, max_width // normalized_ratio]))
        # ranges[i].append(max([max_height // normalized_ratio, max_width]))
        ranges[i].append(max([max_height, max_width // normalized_ratio, crop_size]))
        ranges[i].append(max([max_height // normalized_ratio, max_width, crop_size]))
        centers[i] = [(min_h + max_h) // 2, (min_w + max_w) // 2]
    ranges = np.asarray(ranges).astype(np.int32)
    centers = np.asarray(centers).astype(np.int32)

    return centers, ranges

import torch
from dataset import get_data_transforms, load_data
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import warnings
from sklearn.manifold import TSNE
from PIL import Image
import csv

warnings.simplefilter(action='ignore', category=FutureWarning)

def list_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    #fs : feature_student, ft : feature_teacher
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    #amap_mode = mul : multiple, initialize 1 and multiple, amap_mode = a : add, initialize 0 and add
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
#        print(fs.shape, ft.shape)
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
#        print(a_map.shape)
        a_map = torch.unsqueeze(a_map, dim=1)
#        print(a_map.shape)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
#        print(anomaly_map.shape)
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    
    cam = np.float32(anomaly_map)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def generate_image(image_size, min_index, max_index):
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    
    for i in range(min_index[0], max_index[0] + 1):
        for j in range(min_index[1], max_index[1] + 1):
            image[i, j] = 255

    return image

def evaluation_1(encoder, bn, decoder, dataloader, device):
    bn.eval()
    decoder.eval()
    gt_list = []
    pr_list = []
    sim_list = []
    result_list = []
    cnt = 0
    total = 0
    mean = 0
    with torch.no_grad():
        for img, gt, _, _ in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=0)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            gt_list.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list.extend(anomaly_map.ravel())

        threshold = 0.5  # 이진 분류 문제에서 사용할 임계값 설정
        pr_list_binary = np.array(pr_list) > threshold

        precision = precision_score(gt_list, pr_list_binary)
        accuracy = accuracy_score(gt_list, pr_list_binary)
        recall = recall_score(gt_list, pr_list_binary)
        f1 = f1_score(gt_list, pr_list_binary)
        
    return accuracy, precision, recall, f1


def test(_class_):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(_class_)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './swat/' + _class_
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    
    precision = 1
    recall = 1
    f1 = 1

    return precision, recall, f1

#원본
def evaluation(encoder, bn, decoder, dataloader,device,_class_=None):

    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=0)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))



        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        
    return auroc_px, auroc_sp, round(np.mean(aupro_list),3)


import os

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(256, 256)
    test_path = './swat/' + _class_
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    

    count = 0
    with torch.no_grad():
        for img, gt, label, img_type, img_path in test_dataloader:
            img_filename = os.path.basename(img_path[0])

            decoder.eval()
            bn.eval()

            img = img.to(device)
            
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            
            anomaly_map, amap_list = cal_anomaly_map([inputs[2]], [outputs[2]], img.shape[-1], amap_mode='a')
            original_img_path = img_path[0]
            with Image.open(original_img_path) as original_img:
                original_size = original_img.size  # (Width, Height)

            ano_map_resized = cv2.resize(anomaly_map, original_size, interpolation=cv2.INTER_LINEAR)
            

            ano_map= Image.fromarray(ano_map_resized.astype('float32'), mode='F')


            img = img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255

            img = np.uint8(min_max_norm(img)*255)
            
            output_dir = f'./result/{_class_}'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            tiff_path = f'{output_dir}/{img_filename}_ad.tiff'
            ano_map.save(tiff_path, format='TIFF')

            count = count + 1
            print(count)
           

def vis_nd(name, _class_):
    print(name,':',_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    ckp_path = './checkpoints/' + name + '_' + str(_class_) + '.pth'
    train_dataloader, test_dataloader = load_data(name, _class_, batch_size=16)

    encoder, bn = resnet18(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_resnet18(pretrained=False)
    decoder = decoder.to(device)

    ckp = torch.load(ckp_path)

    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    decoder.eval()
    bn.eval()

    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []

    count = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))


            anomaly_map, amap_list = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'org.png',img)

            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./nd_results/'+name+'_'+str(_class_)+'_'+str(count)+'_'+'ad.png', ano_map)
           
            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map)) 
            
        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1

        ano_score = (prmean_list_sp-np.min(prmean_list_sp))/(np.max(prmean_list_sp)-np.min(prmean_list_sp))
        vis_data = {}
        vis_data['Anomaly Score'] = ano_score
        vis_data['Ground Truth'] = np.array(gt_list_sp)

        with open('vis.pkl','wb') as f:
            pickle.dump(vis_data,f,pickle.HIGHEST_PROTOCOL)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
#        df = pd.concat([df, {"pro": mean(pros), "fpr": fpr, "threshold": th}], ignore_index=True)


    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc

def detection(encoder, bn, decoder, dataloader,device,_class_):
    #_, t_bn = resnet50(pretrained=True)
    bn.load_state_dict(bn.state_dict())
    bn.eval()
    #t_bn.to(device)
    #t_bn.load_state_dict(bn.state_dict())
    decoder.eval()
    gt_list_sp = []
    prmax_list_sp = []
    prmean_list_sp = []
    with torch.no_grad():
        for img, label in dataloader:
#        for img, label, _ in dataloader:

            img = img.to(device)
            if img.shape[1] == 1:
                img = img.repeat(1, 3, 1, 1)
            label = label.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], 'acc')
#            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = gaussian_filter(anomaly_map, sigma=0)


            gt_list_sp.extend(label.cpu().data.numpy())
            prmax_list_sp.append(np.max(anomaly_map))
            prmean_list_sp.append(np.sum(anomaly_map))#np.sum(anomaly_map.ravel().argsort()[-1:][::-1]))

        gt_list_sp = np.array(gt_list_sp)
        indx1 = gt_list_sp == _class_
        indx2 = gt_list_sp != _class_
        gt_list_sp[indx1] = 0
        gt_list_sp[indx2] = 1


        auroc_sp_max = round(roc_auc_score(gt_list_sp, prmax_list_sp), 4)
        auroc_sp_mean = round(roc_auc_score(gt_list_sp, prmean_list_sp), 4)
        
    return auroc_sp_max, auroc_sp_mean
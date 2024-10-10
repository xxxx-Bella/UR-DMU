import torch
from options import *
from config import *
from model import *
import numpy as np
from dataset_loader import *
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")
import wandb


def test(net, config, wandb, test_loader, test_info, step, model_file = None):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        # load_iter = iter(test_loader)
        frame_gt = np.load("frame_label/gt-da.npy")
        frame_predict = []
        
        # cls_label = []
        # cls_pre = []
        # temp_predict = torch.zeros((0)).cuda()
        
        # for i in range(len(test_loader.dataset)):
        #     data, label = next(load_iter)
        #     data = data.cuda()  # torch.Size([1, 31, 10, 2048])
        #     # label = label.cuda()
        
        for i, input in enumerate(test_loader):
            # input = input.to(device) 
            data, label = input 
            data = data.cuda()
            res = net(data)
            # breakpoint()
            # a_predict = res["frame"]
            a_predict = res["frame"].cpu().numpy().mean(0) 

            fpre_ = np.repeat(a_predict, 16)
            frame_predict.append(fpre_)
            # frame_predict = np.concatenate([frame_predict, fpre_]) 

            # temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            # if (i + 1) % 10 == 0 :
            #     # cls_label.append(int(label))
            #     a_predict = temp_predict.mean(0).cpu().numpy()
            #     # cls_pre.append(1 if a_predict.max()>0.5 else 0)  
            #     fpre_ = np.repeat(a_predict, 16)
            #     if frame_predict is None:         
            #         frame_predict = fpre_
            #     else:
            #         frame_predict = np.concatenate([frame_predict, fpre_])  
            #     temp_predict = torch.zeros((0)).cuda()
        
        frame_predict = np.concatenate(frame_predict, axis=0)

        if len(frame_gt) != len(frame_predict):  # [37152, 37056] 11520
            print(f"Error: gt and pred have different lengths: {len(frame_gt)} vs {len(frame_predict)}")
            # if len(frame_gt) > len(frame_predict):
            #     frame_gt = frame_gt[:len(frame_predict)]
            breakpoint()

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
    
        # corrent_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        # accuracy = corrent_num / (len(cls_pre))
        
        precision, recall, th = precision_recall_curve(frame_gt, frame_predict,)
        ap_score = auc(recall, precision)

        wandb.log({
            "roc_auc": auc_score,
            # "accuracy": accuracy,
            "pr_auc": ap_score,
            "scores": frame_predict,
            'roc_curve': wandb.plot.line_series(
                xs=fpr,  
                ys=[tpr], 
                keys=["tpr"], 
                title="ROC Curve", 
                xname="FPR" 
            )})

        test_info["step"].append(step)
        test_info["AUC"].append(auc_score)
        test_info["AP"].append(ap_score)
        # test_info["AC"].append(accuracy)
        
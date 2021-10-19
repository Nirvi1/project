from src.data.fashion import get_data_tes, PATH_DATA_LOCAL

from torch import optim
from src.utils.config import prepare_opt, PATH_SRC_LOCAL
prj_name = 'Resnet34Roi'
opt = prepare_opt(prj_path=PATH_SRC_LOCAL, prj_name=prj_name)

from tqdm import tqdm

dl_tes = get_data_tes(
    data_path=PATH_DATA_LOCAL, 
    batch_size=opt.data.bs_tes,
    img_size=opt.data.img_size)

import torch
y_pred_list = []
y_true_list = []
path = '/ml_model/second/DeepFashion/models/model_90'
from src.model import DensenetRoi, ResNet101RoI

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='dir path')
    args = parser.parse_args()
    path = args.model

    model = ResNet101RoI()
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    from torch.autograd import Variable

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.cuda()
    preds_all = []
    model.eval()
    cc = 0
    correct = []
    t = 0
    with torch.no_grad():
        for d in tqdm(dl_tes):
            if device == 'cpu':
                inputs, lm, targets = Variable(d['img']), Variable(d['landmark']), Variable(d['attr'])
            else:
                inputs, lm = Variable(d['img']).cuda(), Variable(d['landmark']).cuda()
                targets = Variable(d['attr']).cuda()
            outputs = model(inputs, lm)
            # correct = []
            preds = []
            for i in range(6):
                pred = outputs[i].argmax(dim=1, keepdim=True).data.cpu().numpy()
                preds.append(pred)
            preds_all.append(preds)

    res = []
    print(" total ", len(preds_all))
    for i in preds_all:
        for c in range(100):
            items = []
            for r in range(6):
                items.append(i[r][c][0])
            res.append(items)

    import csv

    with open('prediction.txt', 'w') as file:
        for item in res:
            for i in item:
                file.write(str(int(i)) + ' ')
            file.write('\n')

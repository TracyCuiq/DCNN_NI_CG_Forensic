import argparse
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_loader import CGPIM_Data
from model import ResNeXt
from tqdm import tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
from transform import get_transforms
from util import get_optimizer, AverageMeter, save_checkpoint, accuracy
import torchnet.meter as meter
import pandas as pd
from sklearn import metrics



parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='Train')
# datasets 
parser.add_argument('-dataset_path', type=str, default='')
parser.add_argument('-train_txt_path', type=str, default='')
parser.add_argument('-test_txt_path', type=str, default='')
parser.add_argument('-val_txt_path', type=str, default='')
# optimizer
parser.add_argument('--optimizer', default='sgd', choices=['sgd','rmsprop','adam','radam'])
parser.add_argument("--lr",type=float, default=0.001)
parser.add_argument('--lr-fc-times', '--lft', default=5, type=int, metavar='LR', help='initial model last layer rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--no_nesterov', dest='nesterov', action='store_false', help='do not use Nesterov momentum')
parser.add_argument('--alpha', default=0.99, type=float, metavar='M', help='alpha for ')
parser.add_argument('--beta1', default=0.9, type=float, metavar='M', help='beta1 for Adam (default: 0.9)')
parser.add_argument('--beta2', default=0.999, type=float, metavar='M', help='beta2 for Adam (default: 0.999)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# training
parser.add_argument("--checkpoint", type=str, default='./checkpoints')
parser.add_argument("--resume", default='', type=str, metavar='PATH', help='path to save the latest checkpoint')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--start_epoch", default=0, type=int, metavar='N')
parser.add_argument('--epochs', default=30, type=int, metavar='N')
parser.add_argument('--arch', default='resnet50', choices=['resnet34','resnet18','resnet50'])
parser.add_argument('--num_classes', default=2, type=int)
# model path
parser.add_argument('--model_path', default='',type=str)
parser.add_argument('--result_csv', default='')
# util
parser.add_argument('--log_dir', type=str, default='log', help='Name of the log folder')
parser.add_argument('--save_models', type=bool, default=True, help='Set True if you want to save trained models')
parser.add_argument('--pre_trained_model_path', type=str, default=None, help='Pre-trained model path')
parser.add_argument('--pre_trained_model_epoch', type=str, default=None, help='Pre-trained model epoch e.g 200')
parser.add_argument('--num_test_img', type=int, default=16, help='Number of images saved during training')
parser.add_argument('--image_size', type=int, default=224, help='image size')

args = parser.parse_args()



# Use CUDA
torch.cuda.set_device(1)

best_acc = 0

def main():
    global best_acc
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    
    # load data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    # train data
    train_set = CGPIM_Data(root=args.train_txt_path, transform=transformations['val_train'], isTrain=True)
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    # val data
    val_set = CGPIM_Data(root=args.val_txt_path, transform=transformations['val_test'], isTrain=False)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # define model
    model = ResNeXt(2, 3, [3, 4, 6, 3], 2)
    model.cuda()
    
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

    # load checkpoint
    start_epoch = args.start_epoch
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        test_loss, val_acc = val(val_loader, model, criterion, epoch)
        scheduler.step(test_loss)
        print('train_loss: %.3f, val_loss:%.3f, train_acc:%.3f, val_acc:%.3f' % (train_loss, test_loss, train_acc, val_acc) )

        # save_model
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                    'fold': 0,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'train_acc':train_acc,
                    'acc': val_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, single=True, checkpoint=args.checkpoint)

    print("best acc = ", best_acc)


  
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for (inputs, targets) in tqdm(train_loader):
        
        inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy(outputs.data,targets.data)
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(),inputs.size(0))

    return losses.avg, train_acc.avg


def val(val_loader, model, criterion, epoch):
    global best_acc
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()
    confusion_matrix = meter.ConfusionMeter(args.num_classes)
    for _, (inputs, targets) in enumerate(val_loader):
        
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        confusion_matrix.add(outputs.data.squeeze(), targets.long())
        acc1 = accuracy(outputs.data,targets.data)

        losses.update(loss.item(), inputs.size(0))
        val_acc.update(acc1.item(),inputs.size(0))
    return losses.avg,val_acc.avg


def test():
    # data
    transformations = get_transforms(input_size=args.image_size, test_size=args.image_size)
    test_set = CGPIM_Data(root=args.test_txt_path, transform=transformations['test'], isTrain=False)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    # load model
    model = ResNeXt(2, 3, [3, 4, 6, 3], 2)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    # evaluate
    y_pred = []
    y_true = []
    img_paths = []
    with torch.no_grad():
        model.eval()
        for (inputs, targets) in tqdm(test_loader):
            y_true.extend(targets.detach().tolist())
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            probability = torch.max(outputs, dim=1)[1].data.cpu().numpy().squeeze()
            y_pred.extend(probability)

        print("y_pred=", y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("accuracy=", accuracy)
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print("confusion_matrix=", confusion_matrix)
        print(metrics.classification_report(y_true, y_pred))
        # fpr,tpr,thresholds = metrics.roc_curve(y_true,y_pred)
        print("roc-auc score=",metrics.roc_auc_score(y_true, y_pred))
    
        res_dict = {
            'label':y_true,
            'predict':y_pred,

        }
        df = pd.DataFrame(res_dict)
        df.to_csv(args.result_csv, index=False)
        print("write to {args.result_csv} succeed ")

    
if __name__ == "__main__":

    if args.mode == 'Train':
        main()
    else:
        test()
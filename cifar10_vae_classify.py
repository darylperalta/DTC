import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from utils.util import AverageMeter, accuracy
from data.cifarloader import CIFAR10Loader_VAE
from models.resnet_3x3 import ResNet, BasicBlock
from models.vae import VAE, VAE_pred
from torchvision.utils import save_image
import os
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


def plot_tsne_train(eval_loader, model,device, n_clusters=5, filename='tsne.png'):

    torch.manual_seed(1)
    # model = model.to(device)
    # cluster parameter initiate

    # feat_model = nn.Sequential(*list(model.children())[:-1])
    feat_model = model.encoder
    # feat_model = ResNet_features(model)
    feat_model.eval()
    targets = np.zeros(len(eval_loader.dataset))
    feats = np.zeros((len(eval_loader.dataset), 128))
    # for _, ((x, _), label, idx) in enumerate(eval_loader):
    for _, (x, label, idx) in enumerate(eval_loader):

        x = x.to(device)
        feat, _ = feat_model(x)
        # print('feat', feat.shape)
        idx = idx.data.cpu().numpy()
        # feats[idx, :] = feat.data.cpu().numpy()
        # targets[idx] = label.data.cpu().numpy()
        feats[idx, :] = feat.data.cpu().numpy()
        targets[idx] = label.data.cpu().numpy()

    mus_tsne = TSNE(n_components=2).fit_transform(feats)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(mus_tsne)
    x = mus_tsne[:,0]
    y = mus_tsne[:,1]
    data = pd.DataFrame()
    data['x'] = x
    data['y'] = y
    data['label'] = targets
    print('data label')
    print(data['label'])
    print('x', x)
    print('feats', feats)
    # data['label'] = kmeans.labels_
    #current_palette = sns.color_palette()
    #sns.palplot(current_palette)
    ax = sns.scatterplot(
                    x="x", y="y",
                    hue="label",
                    data=data,
                    palette=sns.color_palette("hls", n_clusters),
                    alpha=0.3
                    )
    fig = ax.get_figure()
    fig.savefig(filename)

def get_device(verbose=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #if torch.cuda.device_count() > 1:
    #    print("Available GPUs:", torch.cuda.device_count())
    #    # model = nn.DataParallel(model)
    if verbose:
        print("Device:", device)
    return device

def loss_function(xp, x, mu, logvar, beta=1.0, img_size = 32):
    device = get_device()
    x = x.view(-1, int((img_size**2)*3))
    xp = xp.view(-1, int((img_size**2)*3))
    recon = F.binary_cross_entropy(xp, x, reduction='sum').to(device)
    # recon = nn.BCELoss(size_average=False)(xp, x) / x.size(0)
    #MSE = F.mse_loss(recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kl, recon, kl


def train_classify(model, train_loader, args):
    optimizer = Adam(model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    criterion=nn.CrossEntropyLoss().cuda(device)
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, _) in enumerate(train_loader):
            x, target = x.to(device), label.to(device)
            optimizer.zero_grad()
            output= model(x)
            loss = criterion(output, target)
            acc = accuracy(output, target)
            loss.backward()
            optimizer.step()
            acc_record.update(acc[0].item(), x.size(0))
            loss_record.update(loss.item(), x.size(0))
        print('Train Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))
        test(model, eva_loader, args)
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))

def train(model, train_loader, args):
    device = get_device()
    # print(type(train_loader))
    # (x, label, _) = next(iter(train_loader))
    # print('x', x.shape)
    # x, target = x.to(device), label.to(device)
    # mu, logvar = model.encoder(x)
    # print(mu.shape, logvar.shape)
    # z, mu, logvar = model(x)
    # print(z.shape)
    # (x, label, _) = next(iter(train_loader))
    # print(x)
    # x =
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    # criterion=nn.CrossEntropyLoss().cuda(device)

    for epoch in range(args.epochs):
        # loss_record = AverageMeter()
        # acc_record = AverageMeter()
        recon_losses = AverageMeter()
        kl_losses = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        for batch_idx, (x, label, _) in enumerate(train_loader):
            x, target = x.to(device), label.to(device)
            optimizer.zero_grad()
            xp, mu, logvar = model(x)
            # loss = criterion(output, target)
            # acc = accuracy(output, target)
            # print('x')
            # print(x, x.shape)
            # print(xp, xp.shape)
            # print('end x')
            loss, recon, kl = loss_function(xp, x, mu, logvar, args.beta)
            loss.backward()
            recon_losses.update(recon.float().item()/x.size(0))
            kl_losses.update(kl.float().item()/x.size(0))
            optimizer.step()
            # acc_record.update(acc[0].item(), x.size(0))
            # loss_record.update(loss.item(), x.size(0))
        print('Train Epoch: {} Recon Loss: {:.4f} \t KL Loss: {:.4f}'.format(epoch, recon_losses.avg, kl_losses.avg))
    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))
        # test(model, eva_loader, args)

def test(model, test_loader, args):
    model.eval()
    acc_record = AverageMeter()
    for batch_idx, (x, label, _) in enumerate(test_loader):
        x, target = x.to(device), label.to(device)
        output= model(x)
        acc = accuracy(output, target)
        acc_record.update(acc[0].item(), x.size(0))
    print('Test: Avg Acc: {:.4f}'.format(acc_record.avg))

def test_vae(model, test_loader, args):
    model.eval()
    (x, label, _) = next(iter(test_loader))
    x, target = x.to(device), label.to(device)
    xp, mu, logvar = model(x)
    save_image(x, 'input_vae.png')
    save_image(xp, 'recon_vae.png')

    # acc_record = AverageMeter()
    # for batch_idx, (x, label, _) in enumerate(test_loader):
    #     x, target = x.to(device), label.to(device)
    #     output= model(x)
    #     acc = accuracy(output, target)
    #     acc_record.update(acc[0].item(), x.size(0))
    # print('Test: Avg Acc: {:.4f}'.format(acc_record.avg))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cls',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=5e-03)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', default=180, type=int)
    parser.add_argument('--milestones', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--vae_path', type=str, default='./data/experiments/cifar10_vae/vae_cifar10.pth')
    parser.add_argument('--model_name', type=str, default='vae_cifar10')
    parser.add_argument('--dataset_root', type=str, default='data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= args.exp_root + '{}'.format(runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+args.model_name+'.pth'

    train_loader = CIFAR10Loader_VAE(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = True, aug='once', shuffle=True)
    eva_loader = CIFAR10Loader_VAE(root=args.dataset_root, batch_size=args.batch_size, split='test', labeled = True, aug=None, shuffle=False)
    # model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(device)
    model = VAE(128,32).to(device)
    # train(model, train_loader, args)
    # # (x, label, _) = next(iter(eva_loader))
    #
    # # model.load_state_dict(torch.load(args.model_dir), strict=False)
    # test_vae(model, eva_loader, args)
    #
    # train_loader = CIFAR10Loader_VAE(root=args.dataset_root, batch_size=args.batch_size, split='train',labeled = True, aug=None, shuffle=False)
    # plot_tsne_train(train_loader, model,device, n_clusters=5, filename='tsne_vae.png')


    print('training for classification')
    vae_base = model
    vae_base.load_state_dict(torch.load(args.vae_path), strict=False)
    model = VAE_pred(vae_base, 128,5).to(device)

    train_classify(model,train_loader, args)
    test(model, eva_loader, args)

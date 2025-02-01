import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader, TensorDataset
import sys
from datetime import datetime
import argparse
from collections import Counter
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'util')))
from dataset_loader_7 import MMDataset
from utils import *
from models.transnuseg_7 import TransNuSeg

# On NVIDIA architecture
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# On Apple M chip architecture
#device = torch.device("mps")

print('Using ' + str(device) + ' device')

DATA_PATH = '/home/mateus/MateusPro/dlr_project/cerradata4m_exp/train/'
IMG_SIZE = 128

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def save_training_history(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    '''
    model_type:  default: transnuseg
    alpha: ratio of the loss of nuclei mask loss, dafault=0.3
    beta: ratio of the loss of normal edge segmentation, dafault=0.35
    gamma: ratio of the loss of cluster edge segmentation, dafault=0.35
    sharing_ratio: ratio of sharing proportion of decoders, default=0.5
    random_seed: set the random seed for splitting dataset
    num_epoch: number of epoches
    lr: learning rate
    model_path: if used pretrained model, put the path to the pretrained model here
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, default="transnuseg",
                        help="declare the model type to use, currently only support input being transnuseg")
    parser.add_argument("--alpha", required=True, default=0.3, help="coeffiecient of the weight of nuclei mask loss")
    parser.add_argument("--beta", required=True, default=0.35, help="coeffiecient of the weight of normal edge loss")
    parser.add_argument("--gamma", required=True, default=0.35, help="coeffiecient of the weight of cluster edge loss")
    parser.add_argument("--sharing_ratio", required=True, default=0.5, help=" ratio of sharing proportion of decoders")
    parser.add_argument("--random_seed", required=True, help="random seed")
    parser.add_argument("--batch_size", required=True, help="batch size")
    parser.add_argument("--num_channel", required=True, default=14, help="number of Channels")
    parser.add_argument("--num_classes", required=True, default=7, help="number of classes")
    parser.add_argument("--num_epoch", required=True, help='number of epoches')
    parser.add_argument("--lr", required=True, help="learning rate")
    parser.add_argument("--model_path", default=None, help="the path to the pretrained model")

    args = parser.parse_args()
    model_type = args.model_type
    alpha = float(args.alpha)
    beta = float(args.beta)
    gamma = float(args.gamma)
    sharing_ratio = float(args.sharing_ratio)
    batch_size = int(args.batch_size)
    num_channel = int(args.num_channel)
    num_classes = int(args.num_classes)
    random_seed = int(args.random_seed)
    num_epoch = int(args.num_epoch)
    base_lr = float(args.lr)


    # Model
    model = TransNuSeg(img_size=IMG_SIZE, in_chans=num_channel)
    if args.model_path is not None:
        try:
            model.load_state_dict(torch.load(args.model_path))
        except Exception as err:
            print("{} In Loading previous model weights".format(err))

    model.to(device)

    # Reports
    now = datetime.now()
    create_dir('/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/log')
    logging.basicConfig(filename='/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/log/log_{}_{}.txt'.format(model_type, str(now)), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(
        "Batch size : {} , epoch num: {}, alph: {}, beta : {}, gamma: {}, sharing_ratio = {}".format(batch_size,
                                                                                                     num_epoch, alpha,
                                                                                                     beta, gamma,
                                                                                                     sharing_ratio))
    # Dataset
    total_data = MMDataset(dir_path=DATA_PATH, gpu=device, norm='none') # norm: "none", "0to1", "1to1"
    train_set_size = int(len(total_data) * 0.8)
    val_set_size = len(total_data) - train_set_size
    train_set, val_set = data.random_split(total_data, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(random_seed))
    logging.info("train size {} val size {}".format(train_set_size, val_set_size))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Report
    dataloaders = {"train": trainloader, "val": valloader}
    dataset_sizes = {"train": len(trainloader), "val": len(valloader)}
    logging.info("size train : {}, size val {} ".format(dataset_sizes["train"], dataset_sizes["val"]))

    val_loss = []
    train_loss = []
    lr_lists = []

    # Weights
    class_counter = Counter()
    for _, labels, _ in trainloader:
        mask = labels.view(-1).cpu().numpy()
        class_counter.update(mask.tolist())
    total_pixels = sum(class_counter.values())
    class_frequency = {cls: count / total_pixels for cls, count in class_counter.items()}
    class_weights = {cls: 1.0 / freq for cls, freq in class_frequency.items()}
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    weight_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float).to(device)
    print('Weight tensor: ', weight_tensor)

    # Optimizer
    ce_loss1 = CrossEntropyLoss(weight=weight_tensor)
    #ce_loss1 = CrossEntropyLoss()
    dice_loss1 = DiceLoss(num_classes)
    ce_loss2 = CrossEntropyLoss()
    dice_loss2 = DiceLoss(num_classes)

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    best_loss = 100
    best_epoch = 0

    for epoch in range(num_epoch):
        print(f'====== Epoch {epoch} ======')
        # early stop, if the loss does not decrease for 50 epochs
        if epoch > best_epoch + 30:
            break
        for phase in ['train', 'val']:
            running_loss = 0
            running_loss_wo_dis = 0
            running_loss_seg = 0
            s = time.time()  # start time for this epoch
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            for i, d in enumerate(dataloaders[phase]):
                # Data reading
                sar_opt_img, semantic_mask, edge_mask = d

                # Loading data on device
                sar_opt_img = sar_opt_img.float()
                sar_opt_img = sar_opt_img.to(device)
                semantic_mask = semantic_mask.to(device)
                edge_mask = edge_mask.to(device)

                # Model prediction
                output1, output2 = model(sar_opt_img)

                # Prediction status print
                print('- Pred', np.unique(torch.argmax(output1, dim=1).data.cpu().numpy().ravel()))
                print('- True', np.unique(semantic_mask.data.cpu().numpy().ravel()))

                # Loss
                loss_sem = 0.4 * ce_loss1(output1, semantic_mask.long()) + 0.6 * dice_loss1(output1,semantic_mask.float(), softmax=True)
                loss_edg = 0.4 * ce_loss2(output2, edge_mask.long()) + 0.6 * dice_loss2(output2, edge_mask.float(), softmax=True)

                # Ratio controller
                if epoch < 10:
                    ratio_d = 1
                elif epoch < 20:
                    ratio_d = 0.7
                elif epoch < 30:
                    ratio_d = 0.4
                elif epoch < 40:
                    ratio_d = 0.1
                elif epoch >= 40:
                    ratio_d = 0
                else:
                    ratio_d = 0

                # Calculating total loss
                loss = alpha * loss_sem + beta * loss_edg + ratio_d
                running_loss += loss.item()

                print(f'- Total loss: {loss}')

                # Loss without distillation loss
                running_loss_wo_dis += (alpha * loss_sem + beta * loss_edg).item()
                running_loss_seg += loss_sem.item()

                # Backward
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            e = time.time()
            epoch_loss = running_loss / dataset_sizes[phase]
            # Epoch Loss without distillation loss
            epoch_loss_wo_dis = running_loss_wo_dis / dataset_sizes[phase]
            # Epoch Loss for segmantation
            epoch_loss_seg = running_loss_seg / dataset_sizes[phase]

            logging.info('Epoch {},: loss {}, {},time {}'.format(epoch + 1, epoch_loss, phase, e - s))
            logging.info(
                'Epoch {},: loss without distillation {}, {},time {}'.format(epoch + 1, epoch_loss_wo_dis, phase,
                                                                             e - s))
            logging.info('Epoch {},: loss seg {}, {},time {}'.format(epoch + 1, epoch_loss_seg, phase, e - s))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            if phase == 'val' and epoch_loss_seg < best_loss:
                best_loss = epoch_loss_seg
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                logging.info("Best val loss {} save at epoch {}".format(best_loss, epoch + 1))

    save_training_history(train_loss, val_loss, f'/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/log/historical_transnuseg_7cMSISAR_w_{model_type}.png')
    torch.cuda.empty_cache()

    create_dir('/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/saved_7')
    print('Salvando modelo...')
    torch.save(best_model_wts,
               '/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/saved_7/transnuseg_7cMSISAR_w_epoch:{}_valloss:{}_{}.pt'.format(best_epoch, best_loss, str(now)))
    logging.info(
        'Model saved. at {}'.format(
            '/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/saved_7/transnuseg_7cMSISAR_w_epoch:{}_valloss:{}_{}.pt'.format(best_epoch, best_loss, str(now))))

    model.load_state_dict(best_model_wts)
    model.eval()

    dice_acc_val = 0
    dice_loss_val = DiceLoss(num_classes)

    with torch.no_grad():
        print('Validation...')
        for i, d in enumerate(valloader, 0):
            img, semantic_mask, edge_mask = d

            img = img.float()
            img = img.to(device)

            output1, output2 = model(img)
            d_l = dice_loss_val(output1, semantic_mask.float(), softmax=True)
            dice_acc_val += 1 - d_l.item()

    logging.info("dice_acc {}".format(dice_acc_val / dataset_sizes['val']))


if __name__ == '__main__':
    main()

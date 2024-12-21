import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import argparse

torch.autograd.set_detect_anomaly(True)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_embedding_file', default='cat_seg_512.json', type=str)
    parser.add_argument('--log_dir', type=str, default='cat_seg_512_128')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 128],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[128, 256, 256, 512],
                    )
    args = parser.parse_args()

    num_epochs = args.num_epochs
    os.makedirs(f'ckpt/{args.log_dir}', exist_ok=True)
    train_dataset = Autoencoder_dataset(args.text_embedding_file)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False  
    )
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logdir = f'ckpt/{args.log_dir}'

    argsDict = args.__dict__
    with open(logdir + '_setting.txt', 'w') as f:
        f.writelines('----------start---------' + '\n')
        for key, value in argsDict.items():
            f.writelines(key + ':' + str(value) + '\n')
        f.writelines('----------end---------' + '\n') 
    tb_writer = SummaryWriter(logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):
            data = feature.to("cuda:0")
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)
            tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            tb_writer.add_histogram("feat", outputs, global_iter)

        if epoch > 200:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to("cuda:0")
                with torch.no_grad():
                    outputs = model(data) 
                loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f'ckpt/{args.log_dir}/best_ckpt.pth')
                
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))
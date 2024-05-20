import argparse
from data_preprocess import load_data, get_dataset_dim, accuracy
from models import Orthogonal_Model
import torch
import torch.nn as nn
import numpy as np

def parameter_parser():

    parser = argparse.ArgumentParser(description="My Model.")

    parser.add_argument('--path', default='../data/', type=str)
    parser.add_argument('--dataset', default='EMG', type=str)
    parser.add_argument('--target_domain', default='0', type=str)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--dropout', default=0.4, type=float)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--eval_epoch', default=1, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--hidden', default=512, type=int)
    parser.add_argument('--out_channel', default=32, type=int, help='number of CNN out channel')
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    return parser.parse_args()

def save_emb(features, labels):
    np.save('features.npy', features.cpu().numpy())
    np.save('labels.npy', labels.cpu().numpy())
    print('save done!')

if __name__ == '__main__':
    args = parameter_parser()
    device = torch.device('cuda:{}'.format(args.gpu) if args.gpu >= 0 else 'cpu')

    train_features, train_labels, train_domains, \
    test_features, test_labels, num_class, num_train_domain = load_data(args.dataset, args.path, args.target_domain, device)
    print('Shape: ', train_features.shape, test_features.shape)
    model = Orthogonal_Model(in_channel=train_features.shape[1],
                             out_channel=args.out_channel,
                             n_layers=args.n_layers,
                             mlp_in=get_dataset_dim(args.dataset),
                             mlp_hidden=args.hidden,
                             num_class=num_class,
                             num_domain=num_train_domain,
                             dropout=args.dropout,
                             ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    classifier_loss = nn.CrossEntropyLoss()

    print('Begin Train.')
    best_test_acc = 0.
    best_epoch = 0
    for run in range(args.epoch):
        train_loader = torch.utils.data.DataLoader\
            (torch.arange(0, train_features.shape[0], 1), batch_size=args.batch_size, shuffle=True, drop_last=False)
        model.train()
        loss_avg = []
        i_label_acc = []
        i_domain_acc = []
        r_label_acc = []
        r_domain_acc = []
        for batch in train_loader:
            optimizer.zero_grad()
            temp_features = train_features[batch]
            temp_labels = train_labels[batch]
            temp_domains = train_domains[batch]
            # print(temp_features.shape, temp_labels.shape, temp_domains.shape)
            invariant_class, invariant_domain, relevant_class, relevant_domain = model(temp_features)
            ic_loss = classifier_loss(invariant_class, temp_labels)
            id_loss = classifier_loss(invariant_domain, temp_domains)
            rc_loss = classifier_loss(relevant_class, temp_labels)
            rd_loss = classifier_loss(relevant_domain, temp_domains)
            o_loss = model.orthogonal_loss()

            total_loss = ic_loss + (1 / id_loss) + (1 / rc_loss) + rd_loss + o_loss

            i_label_acc.append(accuracy(invariant_class, temp_labels))
            loss_avg.append(float(total_loss.data))
            total_loss.backward()
            optimizer.step()
        print('Epoch: {:d}, Train Acc: {:.4f}, Loss: {:.4f}'.format(run, np.mean(np.array(i_label_acc)), np.mean(np.array(loss_avg))))
        if run % args.eval_epoch == 0:
            model.eval()
            test_loader = torch.utils.data.DataLoader\
                (torch.arange(0, test_features.shape[0], 1), batch_size=args.batch_size, shuffle=False, drop_last=False)
            all_test = None
            for batch in test_loader:
                with torch.no_grad():
                    temp_features = test_features[batch]
                    invariant_class = model(temp_features)
                    if all_test is None:
                        all_test = invariant_class
                    else:
                        all_test = torch.cat((all_test, invariant_class), dim=0)
            acc = accuracy(all_test, test_labels)
            print('Epoch: {:d}, Test Acc: {:.4f}'.format(run, acc))
            if acc > best_test_acc:
                best_test_acc = acc
                best_epoch = run
                # save_emb(all_test, test_labels)
    print('Best Result in Epoch: {:d} with Acc: {:.4f}'.format(best_epoch, best_test_acc))
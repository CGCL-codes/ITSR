import numpy as np
import torch
from sklearn.metrics import f1_score

def accuracy(output, labels):
    preds = output.max(dim=1)[1].view(-1, 1).cpu()
    correct = labels.cpu()
    micro_f1 = f1_score(correct, preds, average='micro')
    return micro_f1

def get_dataset_dim(dataset):
    dim = 0
    if dataset == 'EMG':
        dim = 416
    elif dataset == 'Oppo':
        dim = 160
    elif dataset == 'UCIHAR':
        dim = 16
    elif dataset == 'Uni':
        dim = 464
    return dim

def process_domain(domains):
    u_domains = torch.unique(domains).numpy()
    to_new_domain = {}
    for i in range(u_domains.shape[0]):
        to_new_domain[u_domains[i]] = i
    domains = domains.numpy()
    for i in range(domains.shape[0]):
        domains[i] = to_new_domain[domains[i]]
    return torch.LongTensor(domains)

def load_data(dataset, path, target_domain, device):
    train_features = None
    train_labels = None
    train_domains = None
    test_features = None
    test_labels = None

    assert dataset in ['EMG', 'UCIHAR', 'Uni', 'Oppo']

    if dataset == 'EMG':
        domains = ['0', '1', '2', '3']
        assert target_domain in domains
        for domain in domains:
            file_name = 'emg_domain_{}.npz'.format(domain)
            domain_data = np.load(path + dataset + '/' + file_name)
            temp_features = torch.FloatTensor(domain_data['features'])
            temp_labels = torch.LongTensor(domain_data['labels'])
            temp_domains = torch.LongTensor(domain_data['domains'])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels
    elif dataset == 'UCIHAR':
        domains = ['0', '1', '2', '3', '4']
        assert target_domain in domains
        for domain in domains:
            file_name = 'ucihar_domain_{}_wd.data'.format(domain)
            domain_data = np.load(path + dataset + '/' + file_name, allow_pickle=True)
            temp_features = torch.FloatTensor(domain_data[0][0])
            temp_labels = torch.LongTensor(domain_data[0][1])
            temp_domains = torch.LongTensor(domain_data[0][2])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels
    elif dataset == 'Uni':
        dataset = 'UniMiB-SHAR'
        domains = ['1', '2', '3', '5']
        assert target_domain in domains
        for domain in domains:
            file_name = 'shar_domain_{}_wd.data'.format(domain)
            domain_data = np.load(path + dataset + '/' + file_name, allow_pickle=True)
            temp_features = torch.FloatTensor(domain_data[0][0])
            temp_features = torch.unsqueeze(temp_features, dim=1)
            temp_labels = torch.LongTensor(domain_data[0][1])
            temp_domains = torch.LongTensor(domain_data[0][2])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels
    elif dataset == 'Oppo':
        domains = ['S1', 'S2', 'S3', 'S4']
        assert target_domain in domains
        dataset = 'Opportunity'
        for domain in domains:
            file_name = 'oppor_domain_{}_wd.data'.format(domain)
            domain_data = np.load(path + dataset + '/' + file_name, allow_pickle=True)
            temp_features = torch.FloatTensor(domain_data[0][0])
            temp_features = torch.unsqueeze(temp_features, dim=1)
            temp_labels = torch.LongTensor(domain_data[0][1])
            temp_domains = torch.LongTensor(domain_data[0][2])
            if domain != target_domain:
                if train_features is None:
                    train_features = temp_features
                    train_labels = temp_labels
                    train_domains = temp_domains
                else:
                    train_features = torch.cat((train_features, temp_features), dim=0)
                    train_labels = torch.cat((train_labels, temp_labels))
                    train_domains = torch.cat((train_domains, temp_domains))
            else:
                test_features = temp_features
                test_labels = temp_labels

    num_class = torch.unique(train_labels).shape[0]
    num_domain = torch.unique(train_domains).shape[0]

    return train_features.to(device), train_labels.to(device), process_domain(train_domains).to(device), \
           test_features.to(device), test_labels.to(device), int(num_class), int(num_domain)
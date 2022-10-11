import time
import torch
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from settings import *

def _train_or_test(model, optimizer=None, dataloader=None):
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_recons_loss = 0
    total_kl_loss = 0
    total_orth_loss = 0

    for i, (image, label) in enumerate(dataloader):
        input = image.to(device)
        target = label.to(device)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, decoded, kl_loss, orth_loss = model(input, label, is_train)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            recons = torch.nn.functional.mse_loss(decoded, input, reduction="mean")
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_recons_loss += recons.item()
            total_kl_loss += kl_loss.item()
            total_orth_loss += orth_loss.item()

        # compute gradient and do SGD step
        if is_train:
            if coefs is not None:
                loss = (coefs['crs_ent'] * cross_entropy
                      + coefs['recon'] * recons
                      + coefs['kl'] * kl_loss
                      + coefs['ortho'] * orth_loss
                        )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del decoded

    end = time.time()

    print('\ttime: \t{0}'.format(end -  start))
    print('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    print('\trecons: \t{0}'.format(total_recons_loss / n_batches))
    print('\tKL: \t{0}'.format(total_kl_loss / n_batches))
    print('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    print('\torth: \t\t{0}'.format(orth_loss / n_batches))

    return n_correct / n_examples, total_cross_entropy/n_batches, total_recons_loss/n_batches, total_kl_loss/n_batches, total_orth_loss/n_batches


def train(model, optimizer=None, dataloader=None):
    assert(optimizer is not None)
    
    print('\ttrain')
    model.train()
    return _train_or_test(model, optimizer=optimizer, dataloader=dataloader)


def test(model, optimizer=None, dataloader=None):
    print('\ttest')
    model.eval()
    return _train_or_test(model, optimizer=optimizer, dataloader=dataloader)



import datetime
import torch.nn.functional as F
import torch


logmsg = ''
timemark = dict()
saveDefault = False


def log(msg, save=None, oneline=False):
    global logmsg
    global saveDefault
    time = datetime.datetime.now()
    tem = '%s: %s' % (time, msg)
    if save != None:
        if save:
            logmsg += tem + '\n'
    elif saveDefault:
        logmsg += tem + '\n'
    if oneline:
        print(tem, end='\r')
    else:
        print(tem)


def marktime(marker):
    global timemark
    timemark[marker] = datetime.datetime.now()


def compute_vgae_loss(logits, adj, norm, vgae_model, weight_tensor):
    vgae_loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
    kl_divergence = 0.5 / logits.size(0) * (
            1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
        1).mean()
    vgae_loss = vgae_loss - kl_divergence
    return vgae_loss


if __name__ == '__main__':
    log('')

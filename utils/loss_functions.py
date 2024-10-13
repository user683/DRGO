import torch
import torch.nn.functional as F


def compute_vgae_loss(logits, adj, norm, vgae_model, weight_tensor):
    vgae_loss = norm * F.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
    kl_divergence = 0.5 / logits.size(0) * (
            1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
        1).mean()
    vgae_loss = vgae_loss - kl_divergence
    return vgae_loss


# bpr_loss
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    user_emb = torch.nan_to_num(user_emb, 0)
    pos_item_emb = torch.nan_to_num(pos_item_emb, 0)
    neg_item_emb = torch.nan_to_num(neg_item_emb, 0)
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb = torch.nan_to_num(emb, 0)
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg



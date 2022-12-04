import torch
import torch.nn as nn

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(self, label, logits, mask):
        # label: [batch_size, conv_len, conv_len]
        # logits: [batch_size, conv_len, conv_len]
        # mask: [batch_size, conv_len, conv_len]
        mask_ = mask.eq(1)
        label = torch.masked_select(label.float(), mask_)
        logits = torch.masked_select(logits, mask_)
        loss = self.loss_fn(logits, label)
        return loss
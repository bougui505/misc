#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Nov 29 16:53:41 2024

import os

import numpy as np
import torch
import torch.nn as nn


def log(msg):
    try:
        logging.info(msg)  # type: ignore
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


class ScaledDotProductAttention(nn.Module):
    """
    See: https://einops.rocks/pytorch-examples.html
    """

    def __init__(self, temperature, attn_dropout=0.1, activation=nn.Softmax(dim=2)):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.activation = activation

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """
    See: https://einops.rocks/pytorch-examples.html
    
    >>> bs = 4
    >>> d_model = 128
    >>> d_k = 16
    >>> d_v = 32
    >>> nq = 100
    >>> nv = 78
    >>> nhead = 8
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v)
    >>> q = torch.rand(bs, nq, d_model)
    >>> v = torch.rand(bs, nv, d_model)
    >>> k = torch.clone(v)
    >>> out, attn =  mha(q, k, v)
    >>> out.shape
    torch.Size([4, 100, 128])
    >>> attn.shape
    torch.Size([4, 100, 78])

    To compare when key_padding_mask is not None
    Put infinite values in k:
    >>> k[bs-1, nv-1] = torch.inf
    >>> k[bs-2, nv-2] = torch.inf
    >>> out, attn =  mha(q, k, v)

    Consequently the output contain nan at those positions
    >>> torch.isnan(out[bs-1, nv-1]).all()
    tensor(True)
    >>> torch.isnan(out[bs-2, nv-2]).all()
    tensor(True)

    and not at the other:
    >>> torch.isnan(out[bs-3, nv-3]).any()
    tensor(False)

    As well as the attention matrix (attn)
    >>> torch.isnan(attn[bs-1, :, nv-1]).all()
    tensor(True)
    >>> torch.isnan(attn[bs-2, :, nv-2]).all()
    tensor(True)
    >>> torch.isnan(attn[bs-3, :, nv-3]).any()
    tensor(False)

    Define a mask
    >>> key_padding_mask = torch.zeros(bs, nv, dtype=bool)
    >>> key_padding_mask[bs-1, nv-1] = True
    >>> key_padding_mask[bs-2, nv-2] = True
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> out.shape
    torch.Size([4, 100, 128])

    The output doesn't contain nan anymore as the infinite values are masked:
    >>> torch.isnan(out[bs-1, nv-1]).any()
    tensor(False)
    >>> torch.isnan(out[bs-2, nv-2]).any()
    tensor(False)

    The attn matrix contain 0 at the masked positions
    >>> attn.shape
    torch.Size([4, 100, 78])
    >>> (attn[bs-1, :, nv-1] == 0).all()
    tensor(True)
    >>> (attn[bs-2, :, nv-2] == 0).all()
    tensor(True)

    The attn matrix is softmaxed
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=0)
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> torch.isclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2))).all()
    tensor(True)

    The user can define another activation function (or identity)
    >>> mha = MultiHeadAttention(n_head=nhead, d_model=d_model, d_k=d_k, d_v=d_v, attn_dropout=0, attn_activation=lambda x: x, attn_temperature=1.0)
    >>> out, attn =  mha(q, k, v, key_padding_mask=key_padding_mask)
    >>> torch.isclose(attn.sum(dim=2), torch.ones_like(attn.sum(dim=2))).all()
    tensor(False)
    
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, attn_dropout=0.1, attn_activation=nn.Softmax(dim=2), attn_temperature=None):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if attn_temperature is None:
            attn_temperature = np.power(d_k, 0.5)
        self.attention = ScaledDotProductAttention(temperature=attn_temperature, attn_dropout=attn_dropout, activation=attn_activation)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None, key_padding_mask=None, average_attn_weights=True):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        residual = q
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Order for n_head and batch size: (n_head, sz_b, ...)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        if key_padding_mask is not None:  #(sz_b, len_k)
            key_padding_mask = torch.stack([key_padding_mask,]*n_head, dim=0).reshape(sz_b*n_head, 1, len_k) * torch.ones(sz_b*n_head, len_q, 1, dtype=torch.bool)
            if mask is not None:
                mask = mask + key_padding_mask
            else:
                mask = key_padding_mask

        output, attn = self.attention(q, k, v, mask=mask)
        # >>> output.shape
        # torch.Size([sz_b, len_q, d_v])
        # >>> attn.shape
        # torch.Size([sz_b*n_head, len_q, len_k])
        if average_attn_weights:
            attn = attn.view(n_head, sz_b, len_q, len_k)
            attn = attn.mean(dim=0)
        
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output, attn


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-a", "--arg1")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )
        sys.exit()

import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F

import src.utils_improvements

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EmbeddingNetv2(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNetv2, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LINEAR_SOFTMAX_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LINEAR_SOFTMAX_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class LINEAR_SOFTMAX(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LINEAR_SOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class LAYER_ALE(nn.Module):
    def __init__(self, input_dim, attri_dim):
        super(LAYER_ALE, self).__init__()
        self.fc = nn.Linear(input_dim, attri_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, attribute):
        batch_size = x.size(0)
        x = torch.mean(x, dim=1)
        x = x.view(batch_size, -1)
        middle = self.fc(x)
        output = self.softmax(middle.mm(attribute))
        return output


class APN(nn.Module):
    def __init__(self, opt):
        super(APN, self).__init__()

        # we left the entry for several layers, but here we only use layer4
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 1*1, 'avg_pool': 1*1}
        if opt.input_size is not None:
            self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2*opt.input_size, 'avg_pool': 2048}
        else:
            self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': opt.input_size_audio + opt.input_size_video,
                                 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 1*1, 'avg_pool': 1}
        self.extract = ['layer4']
        self.epsilon = 1e-4

        self.drop_out_

        self.A_enc = EmbeddingNetv2(input_size=512, hidden_size=512, output_size=300, dropout=0.2, momentum=0.1, use_bn=True)
        self.V_enc = EmbeddingNetv2(input_size=512, hidden_size=512, output_size=300, dropout=0.2, momentum=0.1, use_bn=True)

        self.A_proj = EmbeddingNetv2(input_size=300, hidden_size=512, output_size=512, dropout=0.3, momentum=0.1, use_bn=True)
        self.V_proj = EmbeddingNetv2(input_size=300, hidden_size=512, output_size=512, dropout=0.3, momentum=0.1, use_bn=True)

        self.cross_attention = Transformer(300, 1, 3, 100, 64, dropout=0.1)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 300))

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

        self.prototype_vectors = dict()
        for name in self.extract:
            # 300 * 1024
            prototype_shape = [300, self.channel_dict[name], 1, 1]
            self.prototype_vectors[name] = nn.Parameter(2e-4*torch.rand(prototype_shape), requires_grad=True)
        self.prototype_vectors = nn.ParameterDict(self.prototype_vectors)

        if opt.input_size is not None:
            self.ALE_vector = nn.Parameter(2e-4*torch.rand([300, 2*opt.input_size, 1, 1]), requires_grad=True)
        else:
            self.ALE_vector = nn.Parameter(2e-4*torch.rand([300, opt.input_size_audio + opt.input_size_video, 1, 1]), requires_grad=True)
            
        self.opt = opt

    def forward(self, pos_v, pos_a, attribute, return_map=False):
        """out: predict class, predict attributes, maps, out_feature"""

        phi_a = self.A_enc(pos_a)
        phi_v = self.V_enc(pos_v)

        positive_input = torch.stack((phi_a + self.pos_emb1D[0, :], phi_v + self.pos_emb1D[1, :]), dim=1)
        phi_attn= self.cross_attention(positive_input)
        
        audio_fe_attn = phi_a + phi_attn[:, 0, :]
        video_fe_attn= phi_v + phi_attn[:, 1, :]

        theta_v = self.V_proj(video_fe_attn)
        theta_a = self.A_proj(audio_fe_attn)

        x = torch.cat((theta_v, theta_a), 1)
        # x 256, 1024   attribute 42, 300
        if self.opt.norm_inputs:
            x = F.normalize(x)

        # b, c, 1, 1
        batch_size = x.size(0)
        x=torch.unsqueeze(x,2)
        x=torch.unsqueeze(x,3)

        attention = dict()
        pre_attri = dict()
        pre_class = dict()

        # kernal 300 * 1024 pre_attri['final'] 256*1024 x 1024*300 = 256,300
        pre_attri['final'] = F.max_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=1).view(batch_size, -1)
        
        # 256*300 x 300*42 = 256,42
        output_final = self.softmax(pre_attri['final'].mm(attribute.t()))
        
        for name in self.extract:
            # embedding layer 映射到属性空间 256,1024,1,1 x 1024,300,1,1  = 256, 300,1, 1
            attention[name] = F.conv2d(input=x, weight=self.prototype_vectors[name])
            
            # 属性空间的特征表示 256 x 300
            pre_attri[name] = F.max_pool2d(attention[name], kernel_size=self.kernel_size[name]).view(batch_size, -1)
            # 预测的属性分布 256 x 42
            pre_class[name] = self.softmax(pre_attri[name].mm(attribute.t()))
            
        return output_final, pre_attri, attention, pre_class, attribute

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

    def _l2_convolution(self, x, prototype_vector, one):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2  # [64, C, W, H]
        x2_patch_sum = F.conv2d(input=x2, weight=one)

        p2 = prototype_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototype_vector)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast  [64, 312,  W, H]
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)  # [64, 312,  W, H]
        return distances


class DeviseModel(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    if args.input_size is not None:
        self.bilinear = nn.Linear(self.args.input_size*2, 300, bias=False)
    else:
        self.bilinear = nn.Linear(self.args.input_size_audio + self.args.input_size_video, 300, bias=False)
    self.dropout = nn.Dropout(self.args.dropout_baselines)

  def forward(self, vis_fts, txt_fts):
    if self.args.norm_inputs:
        vis_fts = F.normalize(vis_fts)
    vis_fts = self.dropout(vis_fts)
    txt_fts = self.dropout(txt_fts)
    projected_text_features=self.bilinear(vis_fts)
    logits = torch.matmul(self.bilinear(vis_fts), txt_fts.t())
    return logits, projected_text_features, txt_fts


class CJME(nn.Module):

    def __init__(self, args):
        super(CJME, self).__init__()
        self.dropout = args.dropout_baselines
        self.triplet_net = self._triplet_net(args)
        self.attention_model=nn.Sequential(
                nn.BatchNorm1d(num_features=args.input_size_video+args.input_size_audio),
                nn.Linear(in_features=args.input_size_video+args.input_size_audio, out_features=1),
                nn.Sigmoid()

        )
    def forward(self, x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q):
        a_p, v_p, t_p, a_q, v_q, t_q = self.triplet_net(x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q)
        input_attention=torch.cat((x_a_p, x_v_p), axis=1)
        attention_weights=self.attention_model(input_attention)

        index_video=attention_weights>=0.5
        index_audio=attention_weights<0.5
        threshold_attention=torch.clone(attention_weights)
        threshold_attention[index_video]=1
        threshold_attention[index_audio]=0

        return a_p, v_p, t_p, a_q, v_q, t_q, attention_weights, threshold_attention

    def _triplet_net(self, args):
        if args.input_size is not None:
            f_a = EmbeddingNetCJME(
                input_size=args.input_size,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=self.dropout,
                use_bn=args.embedding_use_bn
            )
            f_v = EmbeddingNetCJME(
                input_size=args.input_size,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=self.dropout,
                use_bn=args.embedding_use_bn
            )
        else:
            f_a = EmbeddingNetCJME(
                input_size=args.input_size_audio,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=self.dropout,
                use_bn=args.embedding_use_bn
            )
            f_v = EmbeddingNetCJME(
                input_size=args.input_size_video,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=self.dropout,
                use_bn=args.embedding_use_bn
            )

        f_t = EmbeddingNetCJME(
            input_size=300,
            hidden_size=300,
            output_size=64,
            dropout=self.dropout,
            use_bn=args.embedding_use_bn
        )
        return TripletNet(f_a, f_v, f_t)

    def get_embedding(self, x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q):
        return self.triplet_net(x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q)

    def get_classes_embedding(self, x_t_p):
        return self.triplet_net.get_classes_embedding(x_t_p)

class EmbeddingNetCJME(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=None):
        super(EmbeddingNetCJME, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.ReLU())
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)




class AVGZSLNet(nn.Module):
    """
    Network class for the whole model. This combines the embedding layers .. math:: F_A, F_V, F_T.
    As well as the cross modal decoder network .. math:: F_{DEC}.
    This calculates two triplets for the positive class p and q, respectively which get fed forward into
    the cross-modal decoder. The relevant data which is needed for the TotalLoss is returned.
    """

    def __init__(self, args):
        super(AVGZSLNet, self).__init__()
        self.triplet_net = self._triplet_net(args)
        self.decoder_net = self._decoder_net(args)

    def forward(self, x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q):
        a_p, v_p, t_p, a_q, v_q, t_q = self.triplet_net(x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q)
        x_ta_p, x_tv_p, x_tt_p, x_ta_q, x_tv_q = self.decoder_net(a_p, v_p, t_p, a_q, v_q)
        return x_t_p, a_p, v_p, t_p, a_q, v_q, t_q, x_ta_p, x_tv_p, x_tt_p, x_ta_q, x_tv_q

    def _triplet_net(self, args):
        if args.input_size is not None:
            f_a = EmbeddingNet(
                input_size=args.input_size,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=args.embedding_dropout,
                use_bn=args.embedding_use_bn
            )
            f_v = EmbeddingNet(
                input_size=args.input_size,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=args.embedding_dropout,
                use_bn=args.embedding_use_bn
            )
        else:
            f_a = EmbeddingNet(
                input_size=args.input_size_audio,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=args.embedding_dropout,
                use_bn=args.embedding_use_bn
            )
            f_v = EmbeddingNet(
                input_size=args.input_size_video,
                hidden_size=args.embeddings_hidden_size,
                output_size=64,
                dropout=args.embedding_dropout,
                use_bn=args.embedding_use_bn
            )

        f_t = EmbeddingNet(
            input_size=300,
            hidden_size=300,
            output_size=64,
            dropout=args.embedding_dropout,
            use_bn=args.embedding_use_bn
        )
        return TripletNet(f_a, f_v, f_t)

    def _decoder_net(self, args):
        f_dec = EmbeddingNet(
            input_size=64,
            hidden_size=args.decoder_hidden_size,
            output_size=300,
            dropout=args.decoder_dropout,
            use_bn=args.decoder_use_bn
        )
        return DecoderNet(f_dec, args.normalize_decoder_outputs)

    def get_embedding(self, x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q):
        return self.triplet_net(x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q)

    def get_classes_embedding(self, x_t_p):
        return self.triplet_net.get_classes_embedding(x_t_p)


class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self, input_size, output_size, hidden_size=None, dropout=0.):
        super(EmbeddingNetL2, self).__init__(input_size, output_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output = F.normalize(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net1, embedding_net2, embedding_net3):
        super(TripletNet, self).__init__()
        self.embedding_net1 = embedding_net1
        self.embedding_net2 = embedding_net2
        self.embedding_net3 = embedding_net3

    def forward(self, x_a_p, x_v_p, x_t_p, x_a_q, x_v_q, x_t_q):

        a_p = self.embedding_net1(x_a_p)
        v_p = self.embedding_net2(x_v_p)
        t_p = self.embedding_net3(x_t_p)
        a_q = self.embedding_net1(x_a_q)
        v_q = self.embedding_net2(x_v_q)
        t_q = self.embedding_net3(x_t_q)

        return a_p, v_p, t_p, a_q, v_q, t_q

    def get_classes_embedding(self, x_t_p):
        return self.embedding_net3(x_t_p)


class DecoderNet(nn.Module):
    def __init__(self, embedding_net, normalize_decoder_outputs):
        super(DecoderNet, self).__init__()
        self.embedding_net = embedding_net
        self.normalize_decoder_outputs = normalize_decoder_outputs

    def forward(self, a_p, v_p, t_p, a_q, v_q):
        x_ta_p = self.embedding_net(a_p)
        x_tv_p = self.embedding_net(v_p)
        x_tt_p = self.embedding_net(t_p)
        x_ta_q = self.embedding_net(a_q)
        x_tv_q = self.embedding_net(v_q)

        if self.normalize_decoder_outputs:
            x_ta_p = F.normalize(x_ta_p)
            x_tv_p = F.normalize(x_tv_p)
            x_tt_p = F.normalize(x_tt_p)
            x_ta_q = F.normalize(x_ta_q)
            x_tv_q = F.normalize(x_tv_q)

        return x_ta_p, x_tv_p, x_tt_p, x_ta_q, x_tv_q

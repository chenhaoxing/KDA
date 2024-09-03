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
import pandas as pd 
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import BertTokenizer, BertModel

import src.utils_improvements
import torch.multiprocessing as mp
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=True)  
        self.b = nn.Parameter(torch.tensor(0.0), requires_grad=True)  
        
    def forward(self, x):
        return x * self.a + self.b

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, features, adjacency_matrix):
        # GCN layer
        x = F.relu(self.fc1(torch.matmul(adjacency_matrix, features)))
        x = self.fc2(torch.matmul(adjacency_matrix, x))
        return x

from InstructorEmbedding import INSTRUCTOR

from transformers import BartTokenizer, BartModel
class KDA(nn.Module):
    def __init__(self, opt):
        super(KDA, self).__init__()
        # we left the entry for several layers, but here we only use layer4
        print(opt)
        self.dim_dict = {'layer1': 56*56, 'layer2': 28*28, 'layer3': 14*14, 'layer4': 1*1, 'avg_pool': 1*1}
        if opt.input_size is not None:
            self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2*opt.input_size, 'avg_pool': 2048}
        else:
            self.channel_dict = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': opt.input_size_audio + opt.input_size_video,
                                 'avg_pool': 2048}
        self.kernel_size = {'layer1': 56, 'layer2': 28, 'layer3': 14, 'layer4': 1*1, 'avg_pool': 1}
        self.extract = ['layer4']
        self.epsilon = 1e-4
        self.drop_rate_enc = 0.
        self.drop_rate_proj = 0.
        self.momentum = 0.1

        self.A_enc = EmbeddingNetv2(input_size=opt.input_size_audio, hidden_size=512, output_size=512, dropout=self.drop_rate_enc, momentum=self.momentum, use_bn=True)
        self.V_enc = EmbeddingNetv2(input_size=opt.input_size_video, hidden_size=1024, output_size=512, dropout=self.drop_rate_enc, momentum=self.momentum, use_bn=True)

        self.A_proj = EmbeddingNetv2(input_size=512, hidden_size=512, output_size=512, dropout=self.drop_rate_proj, momentum=self.momentum, use_bn=True)
        self.V_proj = EmbeddingNetv2(input_size=512, hidden_size=512, output_size=512, dropout=self.drop_rate_proj, momentum=self.momentum, use_bn=True)

        self.cross_attention = Transformer(512, 1, 3, 100, 64, dropout=0.1)
        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 512))

        self.softmax = nn.Softmax(dim=1)
        self.softmax2d = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

        self.ALE_vector = nn.Parameter(2e-4*torch.rand([512, 512*2, 1, 1]), requires_grad=True)

        self.av2com = EmbeddingNetv2(input_size=512, hidden_size=512, output_size=512, dropout=self.drop_rate_enc, momentum=self.momentum, use_bn=True)
        self.t2com = EmbeddingNetv2(input_size=512, hidden_size=512, output_size=512, dropout=self.drop_rate_enc, momentum=self.momentum, use_bn=True)

        self.opt = opt
        files = pd.read_csv( "./avgzsl_benchmark_datasets/VGGSound.csv")
        self.act_name = files["name"].tolist()
        self.act_1= files["description_1"].tolist()
        self.act_2 = files["description_2"].tolist()
        self.act_3 = files["description_3"].tolist()
        
        # self.act_name = [f'A sound event video of "{action}".' for action in self.act_name]
        self.act_name = [f'{action}' for action in self.act_name]

        self.clip = CLIPTextModelWithProjection.from_pretrained('./clip')
        self.clip_tokenizer = AutoTokenizer.from_pretrained('./clip')
        self.clip.requires_grad_(False)

        # self.instructor = INSTRUCTOR('./instructor')
        # self.instructor.requires_grad_(False)

        # self.tokenizer = BartTokenizer.from_pretrained('./bart')
        # self.model = BartModel.from_pretrained('./bart')
        # self.model.requires_grad_(False)

        self.type = 'adaptive_margin'

        if self.type == 'adaptive_margin':
            self.margin_generator = LinearModel()
        elif self.type == 'naive':
            self.margin_generator = None

    def forward(self, pos_v, pos_a, attribute_label, attributes_ids, is_train=True):
        """out: predict class, predict attributes, maps, out_feature"""
        if not is_train:
            import pdb; pdb.set_trace()

        # instruction = "Represent the Action:"
        selected_elements1 = [self.act_1[i] for i in sorted(attributes_ids)]
        text_embeds1 = self.clip.cpu()(**{key: value.to('cpu') for key, value in self.clip_tokenizer(selected_elements1, padding=True, return_tensors="pt").items()}).text_embeds.cuda()

        selected_elements2 = [self.act_2[i] for i in sorted(attributes_ids)]
        text_embeds2 = self.clip.cpu()(**{key: value.to('cpu') for key, value in self.clip_tokenizer(selected_elements2, padding=True, return_tensors="pt").items()}).text_embeds.cuda()

        selected_elements3 = [self.act_3[i] for i in sorted(attributes_ids)]
        text_embeds3 = self.clip.cpu()(**{key: value.to('cpu') for key, value in self.clip_tokenizer(selected_elements3, padding=True, return_tensors="pt").items()}).text_embeds.cuda()

        selected_elements_name = [self.act_name[i] for i in sorted(attributes_ids)]
        text_embeds_name = self.clip.cpu()(**{key: value.to('cpu') for key, value in self.clip_tokenizer(selected_elements_name, padding=True, return_tensors="pt").items()}).text_embeds.cuda()

        text_embeds = torch.stack((text_embeds_name,text_embeds1,text_embeds2,text_embeds3))
        text_embeds = torch.mean(text_embeds, dim=0)
        
        text_embeds_all = [text_embeds[i] for i in attribute_label]
        text_embeds_all = torch.stack(text_embeds_all)
        
        phi_a = self.A_enc(pos_a)
        phi_v = self.V_enc(pos_v)

        positive_input = torch.stack((phi_a + self.pos_emb1D[0, :], phi_v + self.pos_emb1D[1, :]), dim=1)
        phi_attn= self.cross_attention(positive_input)
        
        audio_fe_attn = phi_a + phi_attn[:, 0, :]
        video_fe_attn= phi_v + phi_attn[:, 1, :]

        theta_v = self.V_proj(video_fe_attn)
        theta_a = self.A_proj(audio_fe_attn)

        x = torch.cat((theta_v, theta_a), 1)
        # x = theta_v
        # x 256, 1024   attribute 42, 300
        if self.opt.norm_inputs:
            x = F.normalize(x)

        # av_common = self.av2com(x)
        pre_attri = dict()

        # # b, c, 1, 1
        batch_size = x.size(0)
        x=torch.unsqueeze(x,2)
        x=torch.unsqueeze(x,3)
        pre_attri['final'] = F.max_pool2d(F.conv2d(input=x, weight=self.ALE_vector), kernel_size=1).view(batch_size, -1) 
        output_final = self.softmax(pre_attri['final'].mm(text_embeds.t()))
        
        if is_train:
            b, c = output_final.shape
            if self.type == 'naive':
                margin_scale = -0.3
                add_margin = torch.full((b, c), margin_scale).cuda()
                for i in range(b):
                    add_margin[i][attribute_label[i]] = 0
                output_final = output_final + add_margin

            elif self.type == 'adaptive_margin':
                batch_lenth, class_nums = output_final.shape
                normed_text_embeds = _l2norm(text_embeds)
                similarity_matrix = normed_text_embeds @ torch.transpose(normed_text_embeds, 0, 1)
                
                class_margins = self.margin_generator(similarity_matrix.cuda()) - torch.eye(class_nums).cuda()
                add_margin = torch.zeros(batch_lenth, class_nums).cuda()
                for i in range(b):
                    add_margin[i] = class_margins[attribute_label[i]]
                output_final = output_final + add_margin
            
        return output_final, pre_attri, text_embeds, text_embeds_all


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

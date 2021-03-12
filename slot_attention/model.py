from typing import Tuple

import math
import ipdb
import torch
from torch import nn
from torch.nn import functional as F

from slot_attention.utils import Tensor
from slot_attention.utils import assert_shape
from slot_attention.utils import build_grid
from slot_attention.utils import conv_transpose_out_shape
import numpy as np
import scipy.optimize
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class SlotAttention(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )

    def forward(self, inputs: Tensor):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k.size(), (batch_size, num_inputs, self.slot_size))
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v.size(), (batch_size, num_inputs, self.slot_size))

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots_init = torch.randn((batch_size, self.num_slots, self.slot_size))
        slots_init = slots_init.type_as(inputs)
        slots = self.slots_mu + self.slots_log_sigma.exp() * slots_init

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits = attn_norm_factor * torch.matmul(k, q.transpose(2, 1))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.matmul(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            slots = slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))
            slots = slots + self.mlp(self.norm_mlp(slots))
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))

        return slots


class SlotAttentionModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 64,
        hidden_dims: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_resolution: Tuple[int, int] = (10, 15),
        empty_cache=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.LeakyReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LeakyReLU(),
            nn.Linear(self.out_features, self.out_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution
        out_size = list(in_size)

        for i in range(len(self.hidden_dims) - 1, -1, -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.hidden_dims[i],
                        self.hidden_dims[i - 1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            out_size[0] = conv_transpose_out_shape(out_size[0], 2, 2, 5, 1)
            out_size[1] = conv_transpose_out_shape(out_size[1], 2, 2, 5, 1)
        assert_shape(
            resolution,
            (out_size[0], out_size[1]),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    self.out_features, self.out_features, kernel_size=5, stride=1, padding=2, output_padding=0,
                ),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(self.out_features, 4, kernel_size=3, stride=1, padding=1, output_padding=0,),
            )
        )

        assert_shape(resolution, (out_size[0], out_size[1]), message="")

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, self.decoder_resolution)

        self.slot_attention = SlotAttention(
            in_features=self.out_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=128,
        )
        
        #for p in self.parameters():
        #    p.requires_grad=False

        self.dynamic_pos_embedding = DynamicPositionEmbed(d_model=self.out_features)
        
        self.self_att = SelfAttention(self.out_features, num_attention_heads=1, dropout_prob=0)


        self.slot_pred = nn.Linear(self.out_features*3, self.out_features)
    def forward(self, input):
        #x=input['x']
        #is_pad = input['is_pad']
        x = input.cuda()
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size,max_len,num_channels, height, width = x.shape
        x = x.view(batch_size*max_len,num_channels,height,width)
        #with torch.no_grad():
        encoder_out = self.encoder(x)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out = self.encoder_out_layer(encoder_out)

        slots = self.slot_attention(encoder_out)
        assert_shape(slots.size(), (batch_size*max_len, self.num_slots, self.slot_size))
        slots = slots.view(batch_size,max_len,self.num_slots,self.slot_size)
        batch_size,max_len, num_slots, slot_size = slots.shape
        #slots = self.dynamic_pos_embedding(slots)
        #slots = slots.view(batch_size,max_len*num_slots,slot_size)
        #mask = torch.zeros(batch_size,max_len*num_slots).cuda()
        #for i in range(batch_size):
        #    mask[i][:int(((1-is_pad).sum(dim=1)*num_slots)[i])]=1
        #slots = self.self_att(slots,attention_mask = mask)
        #slots = self.self_att(slots)

        slots = slots.view(batch_size*max_len*num_slots, slot_size, 1, 1)
        decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        #assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))
        slots = slots.view(batch_size,max_len,num_slots,slot_size)
        out = out.view(batch_size,max_len, num_slots, num_channels + 1, height, width)
        recons = out[:,:, :, :num_channels, :, :]
        masks = out[:,:, :, -1:, :, :]
        masks = F.softmax(masks, dim=2)
        recon_combined = torch.sum(recons * masks, dim=2)

	#####
	
        #slots_ = slots.view(batch_size,max_len,self.num_slots,self.slot_size)
	#batch_size,max_len, num_slots, slot_size = slots.shape
        slots_pe,pe = self.dynamic_pos_embedding(slots)
        slots_pe = slots_pe.view(batch_size,max_len*num_slots,slot_size)
        ref_len = max_len-6
        slots_refs = self.self_att(slots_pe[:,:ref_len*num_slots,:],attention_mask=torch.ones(batch_size,ref_len*num_slots).cuda())
        slots_refs = torch.cat((slots_refs,slots_pe[:,:ref_len*num_slots,:]),dim=-1)
        slots_refs = slots_refs.view(batch_size,ref_len,num_slots,slot_size*2)
        slots_ref = slots_refs[:,-1,:,:].unsqueeze(1)
        slots_ref = slots_ref.repeat(1,2,1,1)
        pe_preds = pe[-2:]
        pe_preds = pe_preds.view(2,1,slot_size)
        pe_preds.unsqueeze(0)
        pe_preds = pe_preds.repeat(batch_size,1,num_slots,1)
        slots_preds = self.slot_pred(torch.cat((slots_ref,pe_preds),dim=-1).view(batch_size*2*num_slots,-1))
        #ForkedPdb().set_trace()
        slots_preds = slots_preds.view(batch_size*2*num_slots, slot_size, 1, 1)

        decoder_in = slots_preds.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])
        #with torch.no_grad():

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        #assert_shape(out.size(), (batch_size * num_slots, num_channels + 1, height, width))
        slots_preds = slots_preds.view(batch_size,2,num_slots,slot_size)
        out_preds = out.view(batch_size,2, num_slots, num_channels + 1, height, width)
        recons_preds = out_preds[:,:, :, :num_channels, :, :]
        masks_preds = out_preds[:,:, :, -1:, :, :]
        masks_preds = F.softmax(masks_preds, dim=2)
        recon_combined_preds = torch.sum(recons_preds * masks_preds, dim=2)
            #####
            #ForkedPdb().set_trace()


        return recon_combined, recons, masks, slots, recon_combined_preds, recons_preds, masks_preds, slots_preds



    def loss_function(self, input):
        recon_combined, recons, masks, slots , recon_combined_preds, recons_preds, masks_preds, slots_preds = self.forward(input)
        loss_recon = F.mse_loss(recon_combined, input)
        loss_pred = F.mse_loss(recon_combined_preds, input[:,-2:,:,:,:])
        slots_tgts = slots[:,-2:,:,:].view(-1,7,64)
        slots_preds = slots_preds.view(-1,7,64)
        pairwise_cost = torch.cdist(slots_preds,slots_tgts,p=2)
        indices = np.array(list(map(scipy.optimize.linear_sum_assignment, pairwise_cost.cpu().detach())))
        cost = 0
        for i in range(len(pairwise_cost)):
            cost+=torch.mean(pairwise_cost[i,indices[i][0],indices[i][1]])
        cost /= i
        return {
            "loss_recon": loss_recon,
            "loss_pred": cost+loss_pred,
            "loss":loss_recon+cost+loss_pred,
        }


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj

class DynamicPositionEmbed(nn.Module):

    def __init__(self, d_model):
        super(DynamicPositionEmbed, self).__init__()
        self.d_model=d_model

    def forward(self, x):
        max_len=x.shape[1]
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x=x.permute(0,2,1,3)+pe.cuda()
        x=x.permute(0,2,1,3).contiguous()
        pe = pe.cuda()
        pe.require_grad = False
        return x,pe

class SelfAttention(nn.Module):
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):   
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:   # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads    # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)   
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变
        
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size)    # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.FF = nn.Linear(hidden_size,hidden_size)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)   # 
        return x.permute(0, 2, 1, 3)   # [bs, 8, seqlen, 16]

    def forward(self, hidden_states, attention_mask):
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0   # padding的token置为-10000，exp(-1w)=0
        
        # 线性变换
        mixed_query_layer = self.query(hidden_states)   # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)       # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)   # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)    # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)   # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores + attention_mask
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)    # [bs, 8, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)   # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)   # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        return self.FF(context_layer)    # [bs, seqlen, 128] 得到输出

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

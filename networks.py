import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

class SeparableConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(SeparableConv, self).__init__()
        self.conv1 =nn.Sequential(nn.Conv1d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias),nn.BatchNorm1d(inplanes))
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x.unsqueeze(2))
        return x.squeeze(2)

class FPN_Module(nn.Module):
    def __init__(self, hidden_dim):
        super(FPN_Module, self).__init__()
        self.sub_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, )
        self.sub_conv2 = nn.Conv1d(hidden_dim , hidden_dim // 2, kernel_size=3, padding=1, )
        self.sub_conv3 = nn.Conv1d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, )
        self.s_conv1 = conv_2(1,1)
        self.s_conv2 = conv_2(1,2)
        self.s_conv3 = conv_2(1,4)
        self.up_conv1 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=1,dilation=1)
        self.up_conv2 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=2,dilation=2)
        self.up_conv3 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=4,dilation=4)
        self.up_conv4 = SeparableConv(hidden_dim * 3,hidden_dim,kernel_size=3,padding=8,dilation=8)
        self.out = conv_2(4,1)

    def forward(self, x):
        B,C,N = x.size()
        c1 = self.sub_conv1(x)
        c2 = self.sub_conv2(c1)
        c3 = self.sub_conv3(c2)
        m1 = self.s_conv1(c1.unsqueeze(1)).reshape(B,C,N)
        m2 = self.s_conv2(c2.unsqueeze(1)).reshape(B,C,N)
        m3 = self.s_conv3(c3.unsqueeze(1)).reshape(B,C,N)
        c_out = torch.cat([m1,m2,m3],dim=1)
        d_1 = self.up_conv1(c_out)
        d_2 = self.up_conv2(c_out)
        d_3 = self.up_conv3(c_out)
        d_4 = self.up_conv4(c_out)
        d_out = torch.cat([d_1,d_2,d_3,d_4],dim=1)
        d_out = self.out(d_out.reshape(B,-1,C,N)).reshape(B,C,N)
        return d_out


class MSMA(nn.Module):
    def __init__(self, input_dim,hidden_dim=64, num_classes=256,num_head=4):
        super(MSMA, self).__init__()
        self.num_head = num_head
        self.emb = nn.Sequential(nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=0,),nn.BatchNorm1d(hidden_dim))

        self.ms_f = FPN_Module(hidden_dim)

        self.multi_head = MHA(self.num_head,hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, num_classes)
        self.init_parameters()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, data):
        # Compute 1D convolutional part and apply global max pooling
        x = data.x
        conv_emb = self.emb(x)
        # multi-scale feats
        conv_ms = self.ms_f(conv_emb)
        # attn
        output = self.multi_head(conv_ms)
        # output = torch.flatten(output,1)
        output = self.fc_out(output)
        output = torch.sigmoid(output)
        return output


class MHA(nn.Module):  # multi-head attention

    neigh_k = list(range(3, 21, 2))

    def __init__(self,num_heads,hidden_dim):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.multi_head = nn.ModuleList([
            CPAM(self.neigh_k[i])
            for i in range(num_heads)
        ])
        # self.high_lateral_attn = nn.Sequential(nn.Linear(num_heads * hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,num_heads))
        self.weight_var = Parameter(torch.ones(num_heads))

    def forward(self, x):
        max_pool = self.maxpool(x)
        avg_pool = self.avgpool(x)
        pool_feats = []
        for id,head in enumerate(self.multi_head):
            weight = head(max_pool,avg_pool)
            self_attn = x * weight
            pool_feats.append(torch.max(self_attn,dim=2)[0])

        # concat_pool_features = torch.cat(pool_feats, dim=1)
        # fusion_weights = self.high_lateral_attn(concat_pool_features)
        # fusion_weights = torch.sigmoid(fusion_weights)

        # high_pool_fusion = 0
        # for i in range(self.num_heads):
        #     high_pool_fusion += torch.unsqueeze(fusion_weights[:,i], dim=1) * pool_feats[i]

        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in
                      range(self.num_heads)]
        high_pool_fusion = 0
        for i in range(self.num_heads):
            high_pool_fusion += weight_var[i] * pool_feats[i]

        return high_pool_fusion

class CPAM(nn.Module):
    def __init__(self,k,pool_types = ['avg','max']):
        super(CPAM, self).__init__()
        self.pool_types = pool_types
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, max_pool,avg_pool):
        channel_att_sum = 0.
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                channel_att_raw = self.conv(avg_pool.transpose(1, 2)).transpose(1, 2)
            elif pool_type == 'max':
                channel_att_raw = self.conv(max_pool.transpose(1, 2)).transpose(1, 2)

            channel_att_sum += channel_att_raw

        scale = torch.sigmoid(channel_att_sum)

        return scale

def conv_2(in_planes,out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False), nn.BatchNorm2d(out_planes),nn.ReLU(inplace=True))



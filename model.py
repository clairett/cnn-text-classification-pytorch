import torch
import torch.nn as nn
import torch.nn.functional as F

class  CNN_Text(nn.Module):

    def __init__(self, args, text_field, label_field, word_vector_matrix):
        super(CNN_Text,self).__init__()
        self.args = args
        if text_field is not None:
           self.vocab_stoi = text_field.vocab.stoi
           self.tensor_type = text_field.tensor_type
        if label_field is not None:
           self.label_itos = label_field.vocab.itos

        if word_vector_matrix is not None:
           V = len(word_vector_matrix)
           D = len(word_vector_matrix[0])
        else:
           V = args.embed_num
           D = args.embed_dim

        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.kernel_sizes = Ks

        self.embed = nn.Embedding(V, D)

        if word_vector_matrix is not None:
           self.embed.weight.data.copy_(torch.from_numpy(word_vector_matrix))

           if args.static:
              self.embed.weight.requires_grad = False

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])

        self.dropout = nn.Dropout(args.dropout)

        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc1.weight.data.normal_().mul_(0.01)
        self.fc1.bias.data.zero_()

    def conv_and_pool(self, x, i, conv):
        if x.size(2) < self.kernel_sizes[i]:
           x = nn.ZeroPad2d((0, 0, 0, self.kernel_sizes[i] - x.size(2)))(x)
        x = F.relu(conv(x)).squeeze(3)
        return F.max_pool1d(x, x.size(2)).squeeze(2)

    def forward(self, x):
        x = self.embed(x) # (N,W,D)

        x = x.unsqueeze(1) # (N,Ci,W,D)

        x = [self.conv_and_pool(x, i, conv) \
              for i, conv in enumerate(self.convs1)] #[(N,Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x) # (N,len(Ks)*Co)
        logit = self.fc1(x) # (N,C)

        return logit

    def renorm_fc(self, max_norm):
       self.fc1.weight.data.renorm_(2, 0, max_norm)

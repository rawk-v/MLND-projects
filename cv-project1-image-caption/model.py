import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

        # Add BN layer
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weight()

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)

        # -- Add BN --
        features = self.bn(features)
        return features

    def init_weight(self):
        self.embed.weight.data.normal_(0, 0.02)
        self.embed.bias.data.fill_(0)


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features, captions):
        h = torch.zeros((self.num_layers, captions.shape[0],
                         self.hidden_size), dtype=torch.float32)  # batch_size, num_layers, hidden_size
        c = torch.zeros((self.num_layers, captions.shape[0],
                         self.hidden_size), dtype=torch.float32)

        h = h.cuda()
        c = c.cuda()

        embedded_caption = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embedded_caption), dim=1)
        output, hc = self.lstm(inputs, (h, c))
        return self.fc1(output)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        h = torch.zeros((self.num_layers, inputs.shape[0], self.hidden_size),
                        dtype=torch.float32)  # num_layers, batch_size, hidden_size
        c = torch.zeros((self.num_layers, inputs.shape[0], self.hidden_size), dtype=torch.float32)

        h = h.cuda()
        c = c.cuda()

        predict_word_idx = []

        previous_lstm_output = None
        for i in range(max_len):
            if i == 0:
                embed_input = inputs
            else:
                embed_input = self.embed(previous_lstm_output)
            lstm_output, (h, c) = self.lstm(embed_input, (h, c))
            fc1_output = self.fc1(lstm_output)
            softmax_output = self.softmax(fc1_output)
            top_k = 1
            soft_output_topk = softmax_output.topk(top_k)[1].view(softmax_output.shape[0], top_k)

            previous_lstm_output = soft_output_topk
            idx = soft_output_topk.cpu().view(1).data.numpy()[0]
            predict_word_idx.append(int(idx))
            if idx == 1:  # end index
                print('prediction reached end tag.')
                break
        if len(predict_word_idx) == max_len:
            print('prediction reached max length.')
        return predict_word_idx


if __name__ == '__main__':
    # decoder = DecoderRNN(100, 200, 900, 2)
    encoder = EncoderCNN(100)
    print(encoder)

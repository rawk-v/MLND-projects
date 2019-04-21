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

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features, captions):
#         print("DecoderRNN forward features.shape:", features.shape)
#         print("DecoderRNN forward captions.shape:", captions.shape)
        features = features.view(features.shape[0], 1, features.shape[1])
        h = torch.zeros((self.num_layers, features.shape[0], self.hidden_size), dtype=torch.float32)    # num_layers, batch_size, hidden_size
        c = torch.zeros((self.num_layers, features.shape[0], self.hidden_size), dtype=torch.float32)

        h = h.cuda()
        c = c.cuda()

        softmax_outputs = torch.zeros((features.shape[0], captions.shape[1], self.vocab_size))     # batch_size, n_steps, vocab_size
        softmax_outputs = softmax_outputs.cuda()

        lstm_output, hc = self.lstm(features, (h, c))
        fc1_output = self.fc1(lstm_output)
        softmax_output = self.softmax(fc1_output)   # TODO, save softmax_output
        top_k = 1
        softmax_output_topk = softmax_output.topk(top_k)[1].view(softmax_output.shape[0], top_k)
        embeded_caption = self.embed(softmax_output_topk)
        # print('captions.shape:', captions.shape)
        # print('softmax_outputs.shape:', softmax_outputs.shape)
        # print('fc1_output.shape:', fc1_output.shape)
        # print('softmax_output.shape:', softmax_output.shape)
        # print('softmax_output_topk.shape:', softmax_output_topk.shape)
        # print('embeded_caption.shape:', embeded_caption.shape)

        softmax_outputs[:, 0, :] = softmax_output[:, 0, :]

        for batch in range(features.shape[0]):
            for step in range(captions.shape[1]-1):
                # print('embeded_caption.shape:', embeded_caption.shape, ', step:', step)
                lstm_output, (h[:, batch:batch+1, :], c[:, batch:batch+1, :]) = self.lstm(embeded_caption[batch:batch+1], (h[:, batch:batch+1, :].clone(), c[:, batch:batch+1, :].clone()))
                fc1_output = self.fc1(lstm_output)
                softmax_output = self.softmax(fc1_output)   # TODO, save softmax_output
                # print('softmax_output.shape:', softmax_output.shape)
                # print(softmax_output.topk(1))
                softmax_output_topk = softmax_output.topk(1)[1].view(1, -1)
                embeded_caption[batch:batch+1] = self.embed(softmax_output_topk)

                softmax_outputs[batch, step] = softmax_output[0, 0]

        print("DecoderRNN forward softmax_outputs.shape:", softmax_outputs.shape)
        return softmax_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass


if __name__ == '__main__':
    decoder = DecoderRNN(100, 200, 900, 2)
    print(decoder)

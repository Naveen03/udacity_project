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
        
        self.drop_prob = 0.5
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, dropout=0.1, num_layers=self.num_layers, batch_first=True)
       
        #self.dropout = nn.Dropout(self.drop_prob)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        #self.softmax = nn.Softmax(dim=1)      
    
    def forward(self, features, captions):
        
        ## TODO: Get x, and the new hidden state (h, c) from the lstm
        #print(captions.shape)
        #print(features.shape)
        #print(captions[0])
        
        # batch size
        features = features.unsqueeze(1)
        batch_size = features.shape[0]
        #print('batch_size :' , batch_size)
        #print('batch size2 : ', features.size(0))
        
        h = torch.zeros((self.num_layers, 1, self.hidden_size))
        c = torch.zeros((self.num_layers, 1, self.hidden_size))
        embeded_caption = self.embed(captions[:,:-1])
        
        #print('---Embedded captions---')
        #print('embeded_caption shape: ', embeded_caption.shape)
        #print('features shape : ', features.shape)
        #print('Value: ', embeded_caption )
        
        embeded_caption = torch.cat((features, embeded_caption), dim=1)      
        #print('concatenated shape : ', embeded_caption.shape)
        hiddens, c = self.lstm(embeded_caption)
        out = self.fc(hiddens)
       
        # return x and the hidden state (h, c)
        return out 
    
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    
    def sample(self, inputs, states=None, max_len=5):
    
        """Samples captions for given image features (Greedy search)."""
        h = torch.zeros((self.num_layers, 1, self.hidden_size))
        
        #print('inputs : ', inputs)
        sampled_ids = []
        for i in range(max_len):# maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            # (batch_size, 1, hidden_size), 
            outputs = self.fc(hiddens.squeeze(1))# (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            #predicted = torch.max(outputs, 2)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)# (batch_size, 1, embed_size)
        
        print('sampled_ids : ', sampled_ids)
        sampled_ids = torch.cat(sampled_ids)# (batch_size, 20)
        
        return sampled_ids.squeeze().tolist()
    
    
    
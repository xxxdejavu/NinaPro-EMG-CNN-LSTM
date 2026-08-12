import torch
import torch.nn as nn
import torch.optim as optim

class CNN_BiLSTM_Attention(nn.Module):

    def __init__(self, num_classes):

        super().__init__()


        # CNN feature extractor
        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=10,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(2),


            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(2)

        )


        # BiLSTM
        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=128,

            num_layers=2,

            batch_first=True,

            dropout=0.3,

            bidirectional=True
        )


        # Attention layer
        self.attention = nn.Sequential(

            nn.Linear(256,128),

            nn.Tanh(),

            nn.Linear(128,1)

        )


        # classifier

        self.fc = nn.Sequential(

            nn.Linear(256,64),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(64,num_classes)

        )


    def forward(self,x):


        # x:
        # batch,200,10


        x=x.permute(0,2,1)


        # batch,10,200


        x=self.cnn(x)


        # batch,128,50


        x=x.permute(0,2,1)


        # batch,50,128


        out,_=self.lstm(x)


        # batch,50,256



        # attention score

        score=self.attention(out)


        # batch,50,1



        weight=torch.softmax(
            score,
            dim=1
        )


        # weighted feature

        context=torch.sum(
            weight*out,
            dim=1
        )


        # batch,256


        output=self.fc(context)


        return output

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=CNN_BiLSTM_Attention(23).to(device)
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
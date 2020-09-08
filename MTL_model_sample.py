#MTL用
class RankNet(nn.Module): #MTL
    def __init__(self, n_input, n_out):
        super(RankNet, self).__init__()
        self.n_input = n_input
        mid_io = 128
        self.fc1 = nn.Linear(n_input, mid_io)
        self.fc2 = nn.Linear(mid_io,mid_io)
        #self.fc3 = nn.Linear(128, 2)
        
        self.fc3_1 = nn.Linear(mid_io, 1)
        self.fc3_2 = nn.Linear(mid_io, 1)
        self.fc3_3 = nn.Linear(mid_io, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3_1.weight)
        nn.init.xavier_normal_(self.fc3_2.weight)
        nn.init.xavier_normal_(self.fc3_3.weight)

    def forward(self, x, n_task):
        x = F.relu(self.fc1(x)) # ReLU: max(x, 0)
        x = F.relu(self.fc2(x)) # ReLU: max(x, 0)
        #x = self.fc3(x)
        if(n_task == 0 ): x = self.fc3_1(x)
        if(n_task == 1 ): x = self.fc3_2(x)
        if(n_task == 2 ): x = self.fc3_3(x)
        
        # if(n_task == 'cart'      ): x = self.fc3_1(x)
        # if(n_task == 'click'     ): x = self.fc3_2(x)
        # if(n_task == 'conversion'): x = self.fc3_3(x)

        return x

    def compare(self, x, n_task):
        # m = torch.nn.Softplus()
        # print("compare", x.shape, n_task)
        x1 = x[:,:self.n_input,0]
        x2 = x[:,self.n_input:,0]
        # print("compare2", x1.shape, x2.shape,n_task)
        o = self.forward(x1, n_task) - self.forward(x2, n_task)
        return o
        # return torch.ones(o.shape) / (torch.ones(o.shape) + torch.exp(-o) )

import torch.optim as optim
from moe import *
import util

class Trainer():
    def __init__(self, model: MoE, scaler, lrate, wdecay, clip=3, lr_decay_rate=.97):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaler = scaler
        self.clip = clip
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: lr_decay_rate ** epoch)
        # self.loss_criterion = nn.MSELoss()

    @classmethod
    def from_args(cls, model, scaler, args):
        return cls(model, scaler, args.learning_rate, args.weight_decay, clip=args.clip,
                   lr_decay_rate=args.lr_decay_rate)

    def train(self,adj_mx, input, real_val):
        # #moe
        # real_val = real_val[:, 0, :, :]
        # if real_val.max () == 0: continue
        # #----------------------------
        self.model.train()
        self.optimizer.zero_grad()

        input = input.permute (0, 2, 3, 1)
        output , aux_loss = self.model(adj_mx,input)  # now, output = [batch_size,1,num_nodes, seq_length]
        output = torch.unsqueeze(output,1)
        predict = self.scaler.inverse_transform(output)
        assert predict.shape[1] == 1
        mae, mape, rmse = util.calc_metrics(predict.squeeze(1), real_val, null_val=0.0)
        # loss = self.loss_criterion (output, predict.squeeze(1))
        # total_loss = loss + aux_loss
        # total_loss.backward()
        mae.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        return mae.item(),mape.item(),rmse.item()

    def eval(self,adj_mx, input, real_val):
        self.model.eval()
        input = input.permute (0, 2, 3, 1)
        output , aux_loss = self.model(adj_mx,input) #  [batch_size,seq_length,num_nodes,1]
        output = torch.unsqueeze (output, 1)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        predict = torch.clamp(predict, min=0., max=70.)
        mae, mape, rmse = [x.item() for x in util.calc_metrics(predict, real, null_val=0.0)]
        return mae, mape, rmse

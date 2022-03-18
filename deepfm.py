from multiprocessing.sharedctypes import Value
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.random.manual_seed(2022)

class DeepFM(nn.Module):
    def __init__(self, args) -> None:
        super(DeepFM, self).__init__()
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpuid']

        self.lr = args['lr']
        self.l2_reg = args['l2_reg']
        self.epochs = args['epochs']

        self.num_fetures = args['num_features']
        self.emb_dim = args['embedding_dim']
        self.feature_embs = nn.Embedding(sum(args['field_size']), args['embedding_dim'])
        self.bias_embs = nn.Embedding(sum(args['field_size']), 1)

        # self.num_layers = args['num_layers'] # 2
        self.deep_neurons = args['dense_size']
        self.early_stop = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_norm = args['batch_norm']
        self.opt = args['opt_name']

        if args['deep_layer_act'] == 'relu':
            self.deep_layer_act = nn.ReLU()
        else:
            raise ValueError('Invalid activation function name for deep layers')

        self.dropout_fm_1o = nn.Dropout(p=args['1o_dropout_p'])
        self.dropout_fm_2o = nn.Dropout(p=args['2o_dropout_p'])

        deep_modules = []
        layers_size = [self.num_fetures * self.emb_dim] + args['dense_size']
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            deep_modules.append(nn.Linear(in_size, out_size))
            if self.batch_norm:
                deep_modules.append(nn.BatchNorm1d(num_features=out_size))
            deep_modules.append(self.deep_layer_act)
            deep_modules.append(nn.Dropout(p=args['deep_dropout_p']))
        self.deep = nn.Sequential(*deep_modules)

        self.output = nn.Linear(args['dense_size'][-1] + self.num_fetures + self.emb_dim, 1, bias=False) # concat projection

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.feature_embs.weight, std=0.01)

        glorot = np.sqrt(2.0 / (self.num_fetures * self.emb_dim + self.deep_neurons[0]))

        for la in self.deep:
            if isinstance(la, nn.Linear):
                nn.init.normal_(la.weight, std=glorot)
                nn.init.constant_(la.bias, 0.)
                glorot = np.sqrt(2.0 / (la.weight.size()[0] + la.weight.size()[1]))

        glorot = np.sqrt(2.0 / (self.deep_neurons[-1] + self.num_fetures + self.emb_dim + 1))
        nn.init.normal_(self.output.weight, std=glorot)

    def forward(self, idxs, vals): # idx/vals: batchsize * feature_size
        feat_emb = self.feature_embs(idxs)  # batch_size * feature_size * embedding_size
        feat_emb = torch.multiply(feat_emb, torch.reshape(vals, [feat_emb.size(0), feat_emb.size(1), 1]).expand([feat_emb.size(0), feat_emb.size(1), self.emb_dim])) # batch_size * feature_size * embedding_size
        # first order part
        y_first_order = self.bias_embs(idxs) # batch_size * feature_size * 1
        y_first_order = torch.multiply(y_first_order, torch.reshape(vals, [feat_emb.size(0), feat_emb.size(1), 1])) # batch_size * feature_size * 1
        y_first_order = torch.sum(y_first_order, dim=2) # batch_size * feature_size
        y_first_order = self.dropout_fm_1o(y_first_order) # batch_size * feature_size
        # second order part
        summed_features_emb = torch.sum(feat_emb, 1)  # batch_size * embedding_size
        summed_features_emb_square = torch.square(summed_features_emb) # batch_size * embedding_size
        squared_features_emb = torch.square(feat_emb)  # batch_size * feature_size * embedding_size
        squared_sum_features_emb = torch.sum(squared_features_emb, 1)  # batch_size * embedding_size
        y_second_order = 0.5 * torch.subtract(summed_features_emb_square, squared_sum_features_emb) # batch_size * embedding_size
        y_second_order = self.dropout_fm_2o(y_second_order) # batch_size * embedding_size
        # deep part
        y_deep = feat_emb.view(feat_emb.size(0), -1) # batch_size * (feature_size * embedding_size)
        y_deep = self.deep(y_deep)
        
        concat_input = torch.cat([y_first_order, y_second_order, y_deep], dim=1) # batchsize * (embedding_size+feature_size+last_layer_out_size)
        output = self.output(concat_input)   # batch_size * 1
        out = torch.sigmoid(output)

        return out.view(-1)

    def fit(self, train_loader, valid_loader=None):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()
        if self.opt == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.opt == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=self.lr)
        elif self.opt == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Invalid optimizer name: {self.opt}')

        criterion = nn.BCEWithLogitsLoss(reduction='sum') # CE_log_loss for binary classification
        
        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()
            current_loss = 0.
            total_sample_num = 0
            for labels, idxs, vals in train_loader:
                total_sample_num += labels.size()[0]
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    idxs = idxs.cuda()
                    vals = vals.cuda()
                else:
                    labels = labels.cpu()
                    idxs = idxs.cpu()
                    vals = vals.cpu()
                self.zero_grad()
                # TODO remember to let batches in loader put on GPU or CPU
                prediction = self.forward(idxs, vals)
                loss = criterion(prediction, labels)
                loss += self.l2_reg * self.output.weight.norm()
                for la in self.deep:
                    if la is nn.Linear:
                        loss += self.l2_reg * la.weight.norm()
                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')
                
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

            print(f'[Epoch {epoch:03d}] - training loss={current_loss / total_sample_num:.4f}')
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

            if valid_loader is not None:
                self.eval()
                # TODO if need valdiation
                pass

    def predict(self, test_loader):
        self.eval()
        _, idxs, vals = next(iter(test_loader))
        idxs = idxs.to(self.device)
        vals = vals.to(self.device)
        preds = self.forward(idxs, vals).cpu().detach()

        return preds

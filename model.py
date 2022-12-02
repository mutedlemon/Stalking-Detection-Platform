
import torch
import torch.nn as nn
from transformers import AutoModel


class AttLSTM(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_size: int,
                 hidden_size: int,
                 attention_dim: int,
                 keys: int,
                 fc_dim: int,
                 batch_size: int):
        super(AttLSTM, self).__init__()
        self.keys = keys
        self.fc_dim = fc_dim
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # KcELECTRA 임베딩 레이어
        self.KcELECTRA = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=True)
        self.tanh = nn.Tanh()
        self.attention1 = nn.Linear(2 * self.hidden_size, self.attention_dim, bias=False)
        self.attention2 = nn.Linear(self.attention_dim, self.keys, bias=False)
        self.attention_dist = nn.Softmax(dim=2)

        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))

        # 최종 Morality 예측 레이어
        self.fc1 = nn.Linear(self.keys * self.hidden_size * 2, self.fc_dim, bias=True)
        self.fc3 = nn.Linear(self.fc_dim, 4)
        self.ReLU = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def init__hidden(self, dim):
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, dim, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, dim, self.hidden_size))

    def forward(self, input_ids, token_type_ids, attention_mask):
        # KcELECTRA 임베딩
        self.KcELECTRA.eval()
        with torch.no_grad():
            output = self.KcELECTRA(input_ids=input_ids.squeeze(dim=1),
                                    token_type_ids=token_type_ids.squeeze(dim=1),
                                    attention_mask=attention_mask.squeeze(dim=1))

        output = output[0].squeeze(1)
        output, (h_, c_) = self.LSTM(output, (self.h0, self.c0))

        temp = self.tanh(self.attention1(output))
        score = self.attention2(temp)
        self.A = self.attention_dist(score.transpose(1, 2))
        self.M = self.A.bmm(output)
        self.embedded = self.M.view(self.M.size(0), -1)

        output = self.fc1(self.M.view(self.M.size(0), -1))
        output = self.ReLU(output)
        # output = self.fc2(output)
        # output = self.ReLU(output)
        output = self.fc3(output)

        loss_P = self.penalization(self.A)

        return output, self.embedded, loss_P

    def penalization(self, A):
        eye = torch.eye(A.size(1)).expand(A.size(0), self.keys, self.keys)
        eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5
        loss_P = torch.sum(loss_P) / A.size(0)

        return loss_P


class SurveyEmbed(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_size: int,
                 hidden_size: int,
                 attention_dim: int,
                 keys: int,
                 batch_size: int):
        super(SurveyEmbed, self).__init__()
        self.keys = keys
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # KcELECTRA 임베딩 레이어
        self.KcELECTRA = AutoModel.from_pretrained("beomi/KcELECTRA-base-v2022")

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=True)
        self.tanh = nn.Tanh()
        self.attention1 = nn.Linear(2 * self.hidden_size, self.attention_dim, bias=False)
        self.attention2 = nn.Linear(self.attention_dim, self.keys, bias=False)
        self.attention_dist = nn.Softmax(dim=2)

        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))

    def forward(self, input_ids, token_type_ids, attention_mask):
        # KoBERT 임베딩
        self.KcELECTRA.eval()
        with torch.no_grad():
            output = self.KcELECTRA(input_ids=input_ids.squeeze(dim=1),
                                    token_type_ids=token_type_ids.squeeze(dim=1),
                                    attention_mask=attention_mask.squeeze(dim=1))

        output = output[0].squeeze(1)
        output, (h_, c_) = self.LSTM(output, (self.h0, self.c0))

        temp = self.tanh(self.attention1(output))
        score = self.attention2(temp)
        self.A = self.attention_dist(score.transpose(1, 2))
        self.M = self.A.bmm(output)
        self.embedded = self.M.view(self.M.size(0), -1)

        loss_P = self.penalization(self.A)

        return self.embedded, loss_P

    def penalization(self, A):
        eye = torch.eye(A.size(1)).expand(A.size(0), self.keys, self.keys)
        eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5
        loss_P = torch.sum(loss_P) / A.size(0)

        return loss_P


class SiameseNet(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_size: int,
                 hidden_size: int,
                 attention_dim: int,
                 keys: int,
                 batch_size: int):
        super().__init__()
        self.main_model = SurveyEmbed(num_layers=num_layers,
                                      embedding_size=embedding_size,
                                      hidden_size=hidden_size,
                                      attention_dim=attention_dim,
                                      keys=keys,
                                      batch_size=batch_size)

    def forward_1(self, input_ids, token_type_ids, attention_mask):
        embedded, loss_P = self.main_model(input_ids, token_type_ids, attention_mask)

        return embedded, loss_P

    def forward(self, input_ids_1, token_type_ids_1, attention_mask_1, input_ids_2, token_type_ids_2, attention_mask_2):
        output1, loss_P1 = self.forward_1(input_ids_1, token_type_ids_1, attention_mask_1)
        output2, loss_P2 = self.forward_1(input_ids_2, token_type_ids_2, attention_mask_2)

        return output1, output2, loss_P1, loss_P2


class NewSE(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_size: int,
                 hidden_size: int,
                 attention_dim: int,
                 keys: int,
                 batch_size: int):
        super(NewSE, self).__init__()
        self.keys = keys
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=True)
        self.tanh = nn.Tanh()
        self.attention1 = nn.Linear(2 * self.hidden_size, self.attention_dim, bias=False)
        self.attention2 = nn.Linear(self.attention_dim, self.keys, bias=False)
        self.attention_dist = nn.Softmax(dim=2)

        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))

    def init_hidden(self, dim):
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, dim, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, dim, self.hidden_size))

    def forward(self, BERT_embedding):
        output = BERT_embedding
        output, (h_, c_) = self.LSTM(output, (self.h0, self.c0))

        temp = self.tanh(self.attention1(output))
        score = self.attention2(temp)
        self.A = self.attention_dist(score.transpose(1, 2))
        self.M = self.A.bmm(output)
        self.embedded = self.M.view(self.M.size(0), -1)

        loss_P = self.penalization(self.A)

        return self.embedded, loss_P

    def penalization(self, A):
        eye = torch.eye(A.size(1)).expand(A.size(0), self.keys, self.keys)
        eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5
        loss_P = torch.sum(loss_P) / A.size(0)

        return loss_P


class Stalking(nn.Module):
    def __init__(self,
                 device,
                 model_path: str,
                 batch_size: int):
        super(Stalking, self).__init__()
        self.batch_size = batch_size
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.hardtanh = nn.Hardtanh(max_val=1, min_val=0)

        self.model_1 = NewSE(num_layers=2, embedding_size=768, hidden_size=384, attention_dim=512, keys=30,
                             batch_size=64)
        self.model_2 = NewSE(num_layers=2, embedding_size=768, hidden_size=384, attention_dim=512, keys=30,
                             batch_size=64)
        self.model_3 = NewSE(num_layers=2, embedding_size=768, hidden_size=384, attention_dim=512, keys=30,
                             batch_size=64)
        self.model_4 = NewSE(num_layers=2, embedding_size=768, hidden_size=384, attention_dim=512, keys=30,
                             batch_size=64)
        self.model_5 = NewSE(num_layers=2, embedding_size=768, hidden_size=384, attention_dim=512, keys=30,
                             batch_size=64)
        self.model_10 = NewSE(num_layers=2, embedding_size=768, hidden_size=384, attention_dim=512, keys=30,
                              batch_size=64)

        self.model_1.load_state_dict(torch.load(model_path + '/SE_#1.pt', map_location=device))
        self.model_2.load_state_dict(torch.load(model_path + '/SE_#2.pt', map_location=device))
        self.model_3.load_state_dict(torch.load(model_path + '/SE_#3.pt', map_location=device))
        self.model_4.load_state_dict(torch.load(model_path + '/SE_#4.pt', map_location=device))
        self.model_5.load_state_dict(torch.load(model_path + '/SE_#5.pt', map_location=device))
        self.model_10.load_state_dict(torch.load(model_path + '/SE_#10.pt', map_location=device))

        self.model_1.init_hidden(dim=self.batch_size)
        self.model_2.init_hidden(dim=self.batch_size)
        self.model_3.init_hidden(dim=self.batch_size)
        self.model_4.init_hidden(dim=self.batch_size)
        self.model_5.init_hidden(dim=self.batch_size)
        self.model_10.init_hidden(dim=self.batch_size)

        self.W1 = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)
        self.W2 = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)
        self.W3 = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)
        self.W4 = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)
        self.W5 = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)
        self.W10 = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)

        self.fin_W = nn.Parameter(torch.randn(768, dtype=torch.float32)).to(device)

        self.fc = nn.Linear(768, 1, bias=True)

        nn.init.xavier_uniform_(self.fc.weight)

    def init_hidden(self, dim):
        self.model_1.init_hidden(dim=dim)
        self.model_2.init_hidden(dim=dim)
        self.model_3.init_hidden(dim=dim)
        self.model_4.init_hidden(dim=dim)
        self.model_5.init_hidden(dim=dim)
        self.model_10.init_hidden(dim=dim)

        self.batch_size = dim

    def forward(self, last, reason, action, try_, reaction, relation):
        output1, _1 = self.model_1(last)
        output1 = output1.view(self.batch_size, 30, 768)

        output2, _2 = self.model_2(reason)
        output2 = output2.view(self.batch_size, 30, 768)

        output3, _3 = self.model_3(action)
        output3 = output3.view(self.batch_size, 30, 768)

        output4, _4 = self.model_4(try_)
        output4 = output4.view(self.batch_size, 30, 768)

        output5, _5 = self.model_5(reaction)
        output5 = output5.view(self.batch_size, 30, 768)

        output10, _10 = self.model_10(relation)
        output10 = output10.view(self.batch_size, 30, 768)

        loss_P = _1 + _2 + _3 + _4 + _5 + _10

        self.att1 = self.softmax(torch.matmul(output1, self.W1))
        self.att2 = self.softmax(torch.matmul(output2, self.W2))
        self.att3 = self.softmax(torch.matmul(output3, self.W3))
        self.att4 = self.softmax(torch.matmul(output4, self.W4))
        self.att5 = self.softmax(torch.matmul(output5, self.W5))
        self.att10 = self.softmax(torch.matmul(output10, self.W10))

        q1_att = output1.mul(self.att1.view(self.batch_size, 30, 1)).sum(dim=1)
        q2_att = output2.mul(self.att2.view(self.batch_size, 30, 1)).sum(dim=1)
        q3_att = output3.mul(self.att3.view(self.batch_size, 30, 1)).sum(dim=1)
        q4_att = output4.mul(self.att4.view(self.batch_size, 30, 1)).sum(dim=1)
        q5_att = output5.mul(self.att5.view(self.batch_size, 30, 1)).sum(dim=1)
        q10_att = output10.mul(self.att10.view(self.batch_size, 30, 1)).sum(dim=1)

        total = torch.stack([q1_att, q2_att, q3_att, q4_att, q5_att, q10_att], dim=1)

        self.att_fin = self.softmax(torch.matmul(total, self.fin_W))

        context = total.mul(self.att_fin.view(self.batch_size, 6, 1)).sum(dim=1)

        output = self.fc(context)
        output = self.sigmoid(output)

        return output, loss_P


class NewSE2(nn.Module):
    def __init__(self,
                 num_layers: int,
                 embedding_size: int,
                 hidden_size: int,
                 attention_dim: int,
                 keys: int,
                 fc_dim: int,
                 batch_size: int):
        super(NewSE2, self).__init__()
        self.keys = keys
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        self.hidden_size = hidden_size
        self.fc_dim = fc_dim
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.LSTM = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=True)
        self.tanh = nn.Tanh()
        self.attention1 = nn.Linear(2 * self.hidden_size, self.attention_dim, bias=False)
        self.attention2 = nn.Linear(self.attention_dim, self.keys, bias=False)
        self.attention_dist = nn.Softmax(dim=2)

        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))

        # 최종 Morality 예측 레이어
        self.fc1 = nn.Linear(self.keys * self.hidden_size * 2, self.fc_dim, bias=True)
        self.fc3 = nn.Linear(self.fc_dim, 4)
        self.ReLU = nn.ReLU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def init_hidden(self, dim):
        self.h0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, dim, self.hidden_size))
        self.c0 = torch.nn.parameter.Parameter(torch.zeros(self.num_layers * 2, dim, self.hidden_size))

    def forward(self, BERT_embedding):
        output = BERT_embedding
        output, (h_, c_) = self.LSTM(output, (self.h0, self.c0))

        temp = self.tanh(self.attention1(output))
        score = self.attention2(temp)
        self.A = self.attention_dist(score.transpose(1, 2))
        self.M = self.A.bmm(output)
        self.embedded = self.M.view(self.M.size(0), -1)

        output = self.fc1(self.M.view(self.M.size(0), -1))
        output = self.ReLU(output)
        output = self.fc3(output)

        loss_P = self.penalization(self.A)

        return output, self.embedded, loss_P

    def penalization(self, A):
        eye = torch.eye(A.size(1)).expand(A.size(0), self.keys, self.keys)
        eye = eye.cuda()
        P = torch.bmm(A, A.transpose(1, 2)) - eye
        loss_P = ((P ** 2).sum(1).sum(1) + 1e-10) ** 0.5
        loss_P = torch.sum(loss_P) / A.size(0)

        return loss_P


class Predictor(object):
    def __init__(self,
                 mentality_model,
                 stalking_model,
                 batch_size):
        self.mentality_model = mentality_model
        self.stalking_model = stalking_model
        self.softmax = nn.Softmax()

        self.mentality_model.init_hidden(dim=batch_size)
        self.stalking_model.init_hidden(dim=batch_size)

        self.mentality_model = self.mentality_model.cuda()
        self.stalking_model = self.stalking_model.cuda()

        self.mentality_model.eval()
        self.stalking_model.eval()

    def predict(self, last, reason, action, try_, reaction, charmingCustomer, relation):

        stalk_prob, _ = self.stalking_model(last, reason, action, try_, reaction, relation)
        no1_dep, _1, _ = self.mentality_model(last)
        no2_dan, _1, _ = self.mentality_model(reason)
        no4_anx, _1, _ = self.mentality_model(try_)
        no9_dep, _1, _ = self.mentality_model(charmingCustomer)

        self.result = stalk_prob - self.softmax(no1_dep)[:, 0].unsqueeze(1) - self.softmax(no4_anx)[:, 0].unsqueeze(1) + self.softmax(no2_dan)[:, 0].unsqueeze(1) + self.softmax(no9_dep)[:, 0].unsqueeze(1) + 1.5

        return self.result
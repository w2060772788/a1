import pandas as pd
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
import torch.nn.functional as F


def collate_fn(data):
    ids, texts, labels = [], [], []
    for d in data:
        ids.append(d[0])
        texts.append(d[1])
        labels.append(d[2:7])

    encode = TOKENIZER.batch_encode_plus(
        texts, truncation=True, max_length=512,
        return_tensors="pt", padding=True
    )

    input_ids = encode["input_ids"].to(DEVICE)
    attention_mask = encode["attention_mask"].to(DEVICE)
    # token_type_ids = encode["token_type_ids"].to(DEVICE)
    labels = torch.FloatTensor(labels).to(DEVICE)

    return ids, texts, input_ids, attention_mask, labels


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.pre_train = PRE_TRAIN
        self.linear = nn.Linear(in_features=PRE_TRAIN.config.hidden_size, out_features=5)
        nn.init.xavier_normal_(self.linear.weight.data)

    def forward(self, input_ids, attention_mask):
        cls_emb = self.pre_train(input_ids, attention_mask)[0][:, 0, :]
        linear_out = self.linear(cls_emb)  # [None, 5]
        return linear_out  # [None, 5]


def model_loss(y_true, y_pred, classifier):
    criterion = nn.BCEWithLogitsLoss()
    loss1 = criterion(y_pred, y_true)
    weight = classifier
    weight = weight / torch.norm(weight, 2, 1, keepdim=True)
    inner_pro = torch.cdist(weight, weight)
    inner_pro = torch.triu(inner_pro, diagonal=1)
    pro_mask = inner_pro > 0
    weight_wise = torch.mean(1. / (inner_pro[pro_mask] * inner_pro[pro_mask]))
    return loss1 + weight_wise


def kd_loss(prob_s, prob_t):
    KLDiv = nn.KLDivLoss(reduction="batchmean")
    s_prob = F.log_softmax(prob_s / 2, dim=1)
    t_prob = F.softmax(prob_t / 2, dim=1)
    loss = (1 ** 2) * KLDiv(s_prob, t_prob)

    return loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained("bert_base_uncased")
PRE_TRAIN = BertModel.from_pretrained("bert_base_uncased")

train = pd.read_csv('data/train/eng.csv')
dev = pd.read_csv('data/dev/eng.csv')
train_data = train.values.tolist()
dev_data = dev.values.tolist()

train_loader = DataLoader(train_data, shuffle=True, drop_last=False, batch_size=32, collate_fn=collate_fn)
dev_loader = DataLoader(dev_data, shuffle=False, drop_last=False, batch_size=128, collate_fn=collate_fn)

epochs = 100
warm = 15
t_model = Model().to(DEVICE)
s_model = Model().to(DEVICE)
new_model = Model().to(DEVICE)
optimizer = Adam(s_model.parameters(), lr=3e-5, weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    s_model.train()
    train_loss = 0
    for _, _, input_ids, attention_mask, labels in train_loader:
        prob = s_model(input_ids, attention_mask)
        prob_t = t_model(input_ids, attention_mask)
        loss = model_loss(labels, prob, s_model.linear.weight)
        if epoch >= warm:
            loss += kd_loss(prob, prob_t)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"epoch: {epoch + 1}, loss: {train_loss}")
    scheduler.step()

    if (epoch + 1) % warm == 0:
        s_model.eval()
        predictions = []
        with torch.no_grad():
            for ids, texts, input_ids, attention_mask, _ in dev_loader:
                prob = s_model(input_ids, attention_mask)
                prob = torch.sigmoid(prob)
                pred = (prob > 0.5).int()

                for i in range(len(ids)):
                    prediction = {
                        'id': ids[i],
                        'text': texts[i],
                        'anger': pred[i, 0].item(),
                        'fear': pred[i, 1].item(),
                        'joy': pred[i, 2].item(),
                        'sadness': pred[i, 3].item(),
                        'surprise': pred[i, 4].item()
                    }
                    predictions.append(prediction)

        df_predictions = pd.DataFrame(predictions)
        pre_data = df_predictions.values.tolist()
        combined_data = train_data + pre_data
        train_loader = DataLoader(combined_data, shuffle=True, drop_last=False, batch_size=32, collate_fn=collate_fn)

        t_model.load_state_dict(s_model.state_dict())
        s_model.load_state_dict(new_model.state_dict())

    # model.eval()
    # all_pred, all_label = [], []
    # with torch.no_grad():
    #     for _, _, input_ids, attention_mask, labels in train_loader:
    #         prob = model(input_ids, attention_mask)
    #         all_pred.append(prob.cpu())
    #         all_label.append(labels.cpu())
    #
    # all_pred = torch.cat(all_pred, dim=0)
    # all_label = torch.cat(all_label, dim=0)
    # f1_val = f1_score(all_label.numpy(), (all_pred.numpy() > 0.5).astype(int), average="macro")
    # print(f"[evaluation] epoch: {epoch + 1}, f1: {f1_val}")
    print()

t_model.eval()
predictions = []
with torch.no_grad():
    for ids, _, input_ids, attention_mask, _ in dev_loader:
        prob = t_model(input_ids, attention_mask)
        prob = torch.sigmoid(prob)

        pred = (prob > 0.5).int()

        for i in range(len(ids)):
            prediction = {
                'id': ids[i],
                'anger': pred[i, 0].item(),
                'fear': pred[i, 1].item(),
                'joy': pred[i, 2].item(),
                'sadness': pred[i, 3].item(),
                'surprise': pred[i, 4].item()
            }
            predictions.append(prediction)

df_predictions = pd.DataFrame(predictions)
df_predictions.to_csv('pred_eng.csv', index=False)

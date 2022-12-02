
import torch
import pandas as pd
from tqdm import tqdm
from preprocess import pos_neg


class WellSet(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 text_column: str,
                 data: pd.DataFrame,
                 shuffle: bool):
        self.data = data
        self.tokenizer = tokenizer

        dataset = []

        if shuffle:
            self.data.sample(frac=1).reset_index(drop=True)

        for i in tqdm(range(self.data.shape[0])):
            tokenized = tokenizer.encode_plus(self.data[text_column][i], max_length=512, padding='max_length',
                                              truncation=True, return_tensors='pt')
            input_ids = tokenized['input_ids']
            token_type_ids = tokenized['token_type_ids']
            attention_mask = tokenized['attention_mask']

            if self.data['intent'][i] == 'Depression':
                target = 0
            elif self.data['intent'][i] == 'Shared':
                target = 1
            elif self.data['intent'][i] == 'Anxiety':
                target = 2

            dataset.append((input_ids, token_type_ids, attention_mask, target))

        self.dataset = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        result = {
            'input_ids': self.dataset[item][0],
            'token_type_ids': self.dataset[item][1],
            'attention_mask': self.dataset[item][2],
            'target': torch.as_tensor(self.dataset[item][3], dtype=torch.int32)
        }

        return result


# Siamese Network Dataset

class SiameseSet(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 text_column: str,
                 data: pd.DataFrame,
                 n_pairing: int,
                 shuffle: bool):
        self.data = pos_neg(data=data, context=text_column, n_pairing=n_pairing)
        self.tokenizer = tokenizer

        dataset = []

        if shuffle:
            self.data.sample(frac=1).reset_index(drop=True)

        for i in tqdm(range(self.data.shape[0])):
            tokenized_1 = tokenizer.encode_plus(self.data['First'][i], max_length=512, padding='max_length',
                                                truncation=True, return_tensors='pt')
            tokenized_2 = tokenizer.encode_plus(self.data['Second'][i], max_length=512, padding='max_length',
                                                truncation=True, return_tensors='pt')

            input_ids_1 = tokenized_1['input_ids']
            token_type_ids_1 = tokenized_1['token_type_ids']
            attention_mask_1 = tokenized_1['attention_mask']

            input_ids_2 = tokenized_2['input_ids']
            token_type_ids_2 = tokenized_2['token_type_ids']
            attention_mask_2 = tokenized_2['attention_mask']

            target = self.data['Is_Same'][i]

            dataset.append((input_ids_1, token_type_ids_1, attention_mask_1,
                            input_ids_2, token_type_ids_2, attention_mask_2,
                            target))

        self.dataset = dataset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        result = {
            'input_ids_1': self.dataset[item][0],
            'token_type_ids_1': self.dataset[item][1],
            'attention_mask_1': self.dataset[item][2],
            'input_ids_2': self.dataset[item][3],
            'token_type_ids_2': self.dataset[item][4],
            'attention_mask_2': self.dataset[item][5],
            'target': torch.as_tensor(self.dataset[item][6], dtype=torch.int32)
        }

        return result


# Stalking Classifier Dataset

class STALK(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 embed_model,
                 device,
                 data: pd.DataFrame,
                 shuffle: bool):
        self.data = data.dropna(axis=0).reset_index(drop=True)
        self.embed_model = embed_model
        self.tokenizer = tokenizer

        # self.embed_model = self.embed_model.cuda()
        self.embed_model.eval()

        dataset = []

        if shuffle:
            self.data.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):

        temp = []

        for j in ['last', 'reason', 'action', 'try', 'reaction', 'relation']:
            tokenized = self.tokenizer.encode_plus(self.data.loc[item, j], max_length=512, padding='max_length',
                                              truncation=True, return_tensors='pt')
            tokenized = tokenized.to(device)

            with torch.no_grad():
                embed = self.embed_model(input_ids=tokenized['input_ids'],
                                         token_type_ids=tokenized['token_type_ids'],
                                         attention_mask=tokenized['attention_mask'])

                temp.append(embed[0].squeeze(0))

            del tokenized

        temp.append(self.data.loc[item, 'warning'])

        result = {
            'last': temp[0].cpu().detach(),
            'reason': temp[1].cpu().detach(),
            'action': temp[2].cpu().detach(),
            'try': temp[3].cpu().detach(),
            'reaction': temp[4].cpu().detach(),
            'relation': temp[5].cpu().detach(),
            'target': torch.as_tensor(temp[6], dtype=torch.int32)
        }

        return result


class Tester(torch.utils.data.Dataset):
    def __init__(self,
                 tokenizer,
                 embed_model,
                 data: pd.DataFrame):
        self.data = data
        self.embed_model = embed_model
        self.tokenizer = tokenizer

        self.embed_model.eval()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        temp = []

        for j in ['last', 'reason', 'action', 'try', 'reaction', 'charmingCustomer', 'relation']:
            tokenized = self.tokenizer.encode_plus(self.data.loc[item, j], max_length=512, padding='max_length',
                                                   truncation=True, return_tensors='pt')
            tokenized = tokenized.to(device)

            with torch.no_grad():
                embed = self.embed_model(input_ids=tokenized['input_ids'],
                                         token_type_ids=tokenized['token_type_ids'],
                                         attention_mask=tokenized['attention_mask'])

                temp.append(embed[0].squeeze(0))

            del tokenized

        temp.append(self.data.loc[item, 'warning'])

        result = {
            'last': temp[0].cpu().detach(),
            'reason': temp[1].cpu().detach(),
            'action': temp[2].cpu().detach(),
            'try': temp[3].cpu().detach(),
            'reaction': temp[4].cpu().detach(),
            'charmingCustomer': temp[5].cpu().detach(),
            'relation': temp[6].cpu().detach(),
            'target': torch.as_tensor(temp[7], dtype=torch.int32)
        }

        return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

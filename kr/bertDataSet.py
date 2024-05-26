import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, tokenizer, max_len, pad=True, pair=False):
        self.sentences = []
        self.labels = []

        self.tokenizer = tokenizer
        self.max_len = max_len

        for i in dataset:
            encoding = self.tokenizer.encode_plus(
                text=i[sent_idx],
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length' if pad else None,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.sentences.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten() if 'token_type_ids' in encoding else torch.zeros(self.max_len)
            })

            self.labels.append(torch.tensor(int(i[label_idx]), dtype=torch.long))

    def __getitem__(self, idx):

        return {
            'input_ids': self.sentences[idx]['input_ids'],
            'attention_mask': self.sentences[idx]['attention_mask'],
            'token_type_ids': self.sentences[idx]['token_type_ids'],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

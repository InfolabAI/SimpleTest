from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertModel
import math
import torch


class HandleTransformer:
    def __init__(self):
        pass

    @classmethod
    def get_bert(self, model_name, max_seq_length):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        return tokenizer, model

    def load_positional_encoding(self, dim_feature=1, max_position=1000):
        """
        feature 의 위치를 추론에 포함하기 위해 positional embedding을 계산
        https://github.com/InfolabAI/References/blob/eef3666c88f9c4eb5117a0425652295eca012b0e/models/nezha/modeling_nezha.py#L154

        Args:
            d_model: feature의 dimension (현재 1)
            max_len: 위치의 최대값 (현재 window size)
        """
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_position, dim_feature).float()
        pe.require_grad = False

        position = torch.arange(0, max_position).float().unsqueeze(1)
        div_term = (
            torch.arange(0, dim_feature, 2).float() * -(math.log(10000.0) / dim_feature)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe

    def build_an_input_example(self, indices):
        """
        위치에 맞는 positional_embedding 을 return

        Args:
            indices: 'outlier_label' 열이 포함되지 않은 dataframe
        """
        return (
            self.pe.gather(0, torch.LongTensor(indices).unsqueeze(1))
            .numpy()
            .reshape(-1)
        )

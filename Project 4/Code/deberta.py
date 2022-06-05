from torch import nn
from transformers import DebertaConfig, DebertaModel


class OrcDeBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(2, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 128))

        config = DebertaConfig(max_position_embeddings = 300, hidden_size = 128, num_hidden_layers = 8,
                               num_attention_heads = 4, intermediate_size = 512)
        self.deberta = DebertaModel(config)

        self.reconstruction = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 5))

    def forward(self, inputs, mask):
        embedding = self.embedding(inputs[0:2, :])
        feature = self.deberta(attention_mask = mask, token_type_ids = inputs[:, 2],
                               inputs_embeds = embedding, return_dict = False)
        outputs = self.reconstruction(feature)
        return outputs


if __name__ == "__main__":
    augmentor = OrcDeBERTa()

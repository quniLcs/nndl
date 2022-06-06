from torch import nn
from transformers import DebertaConfig, DebertaModel


class OrcDeBERTa(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU())

        config = DebertaConfig(max_position_embeddings = 300, hidden_size = 128, num_hidden_layers = 8,
                               num_attention_heads = 4, intermediate_size = 512, type_vocab_size = 2,
                               position_biased_input = False, relative_attention = True, max_relative_positions = 8)
        self.deberta = DebertaModel(config)

        self.reconstruction = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 5), nn.GELU())

        self.connection = nn.Linear(1500, 2500)

    def forward(self, inputs, mask):
        # inputs.shape = [batch_size, 300, 4]
        # mask.shape = [batch_size, 300]
        # embedding.shape = [batch_size, 300, 128]
        # feature.shape = [batch_size, 300, 128]
        # feature.shape = [batch_size, 300, 5]
        # outputs.shape = [batch_size, 50, 50]
        embedding = self.embedding(inputs[:, :, 0:3].float())
        feature = self.deberta(attention_mask = mask, token_type_ids = inputs[:, :, 3].int(),
                               inputs_embeds = embedding, return_dict = False)[0]
        feature = self.reconstruction(feature)
        outputs = self.connection(feature.reshape((-1, 1500))).reshape((-1, 50, 50))
        return outputs

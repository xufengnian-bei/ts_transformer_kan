import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.kan_layer import NaiveFourierKANLayer


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity + kan
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

        self.kan1 = NaiveFourierKANLayer(512, 512, gridsize=300)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        #  x_enc = batch_x                 torch.Size([32, 96, 1])
        # x_mark_enc = batch_x_mark        torch.Size([32, 96, 4])
        #  x_dec = dec_inp                 torch.Size([32, 144, 1])
        # x_mark_dec = batch_y_mark        torch.Size([32, 144, 4])


        # enc_out = self.enc_embedding(x_enc, x_mark_enc)                    # torch.Size([32, 96, 512])

        batch_size = x_enc.size(0)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        reshaped_enc_out = enc_out.reshape(-1, 512)
        kan_enc_out = self.kan1(reshaped_enc_out)
        enc_out = kan_enc_out.reshape(batch_size, -1, 512)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)    # torch.Size([32, 96, 512]), 

        dec_out = self.dec_embedding(x_dec, x_mark_dec)                    # torch.Size([32, 144, 512])
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) # torch.Size([32, 144, 7])
 
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

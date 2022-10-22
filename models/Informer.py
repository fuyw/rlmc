import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(c_in=configs.enc_in,
                                           d_model=configs.d_model,
                                           embed_type=configs.embed,
                                           freq=configs.freq,
                                           dropout=configs.dropout)
        self.dec_embedding = DataEmbedding(c_in=configs.dec_in,
                                           d_model=configs.d_model,
                                           embed_type=configs.embed,
                                           freq=configs.freq,
                                           dropout=configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=ProbAttention(
                            mask_flag=False,
                            factor=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for _ in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    self_attention=AttentionLayer(
                        attention=ProbAttention(mask_flag=True,
                                                factor=configs.factor,
                                                attention_dropout=configs.dropout,
                                                output_attention=False),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads),
                    cross_attention=AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False,
                            scale=configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False),

                        # TODO: Performance drops when using ProbAttention
                        # attention=ProbAttention(
                        #     mask_flag=False,
                        #     factor=configs.factor,
                        #     attention_dropout=configs.dropout,
                        #     output_attention=False),

                        d_model=configs.d_model,
                        n_heads=configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

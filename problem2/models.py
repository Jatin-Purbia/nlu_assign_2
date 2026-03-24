import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn          = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.drop         = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        embeds = self.drop(self.embedding(inputs))          # (B, T, E)
        output, _ = self.rnn(embeds)                        # (B, T, H)
        return self.output_layer(self.drop(output))         # (B, T, V)

    def generate_step(self, x, h):
        # x: (1, 1)  h: (1, 1, H)
        embed = self.embedding(x.squeeze(1))                # (1, E)
        output, h_new = self.rnn(embed.unsqueeze(1), h)     # output: (1,1,H)
        return self.output_layer(output.squeeze(1)), h_new  # logits: (1,V), h: (1,1,H)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder      = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.bridge_h     = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_c     = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder_cell = nn.LSTMCell(embed_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.drop         = nn.Dropout(dropout)

    def _encode(self, embeds):
        # h_n, c_n: (2, B, H)
        _, (h_n, c_n) = self.encoder(embeds)
        h = torch.tanh(self.bridge_h(torch.cat([h_n[0], h_n[1]], dim=1)))  # (B, H)
        c = torch.tanh(self.bridge_c(torch.cat([c_n[0], c_n[1]], dim=1)))  # (B, H)
        return h, c

    def forward(self, inputs, lengths):
        embeds = self.drop(self.embedding(inputs))
        h, c   = self._encode(embeds)
        logits_list = []
        for t in range(inputs.shape[1]):
            h, c = self.decoder_cell(embeds[:, t, :], (h, c))
            logits_list.append(self.output_layer(self.drop(h)).unsqueeze(1))
        return torch.cat(logits_list, dim=1)

    def generate_step(self, x, h, c):
        embed = self.embedding(x.squeeze(1))
        h_new, c_new = self.decoder_cell(embed, (h, c))
        return self.output_layer(h_new), h_new, c_new

    def get_initial_decoder_state(self, inputs, lengths, device):
        if inputs is None or inputs.numel() == 0:
            return (torch.zeros(1, self.hidden_size, device=device),
                    torch.zeros(1, self.hidden_size, device=device))
        embeds = self.embedding(inputs)
        return self._encode(embeds)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BahdanauAttention(nn.Module):
    """Additive attention: score = V_a · tanh(W_a·enc_h + U_a·dec_h)"""

    def __init__(self, hidden_size: int, attn_size: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, attn_size, bias=False)
        self.U_a = nn.Linear(hidden_size, attn_size, bias=False)
        self.V_a = nn.Linear(attn_size, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden, mask=None):
        # enc_outputs: (B, S, H), dec_hidden: (B, H)
        energy = self.V_a(torch.tanh(
            self.W_a(enc_outputs) + self.U_a(dec_hidden).unsqueeze(1)
        )).squeeze(-1)                                       # (B, S)
        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(energy, dim=1)
        context = (attn_weights.unsqueeze(1) @ enc_outputs).squeeze(1)  # (B, H)
        return context, attn_weights


class AttentionRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 attn_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder      = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.attention    = BahdanauAttention(hidden_size, attn_size)
        self.decoder_cell = nn.RNNCell(embed_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.drop         = nn.Dropout(dropout)

    def _encode(self, embeds):
        enc_outputs, h_n = self.encoder(embeds)   # (B, T, H), (1, B, H)
        return enc_outputs, h_n.squeeze(0)        # h_dec: (B, H)

    def forward(self, inputs, lengths):
        embeds = self.drop(self.embedding(inputs))
        enc_outputs, h_dec = self._encode(embeds)
        pad_mask = (inputs == 0)
        logits_list = []
        for t in range(inputs.shape[1]):
            context, _ = self.attention(enc_outputs, h_dec, mask=pad_mask)
            h_dec = self.decoder_cell(torch.cat([embeds[:, t, :], context], dim=1), h_dec)
            logits_list.append(self.output_layer(self.drop(h_dec)).unsqueeze(1))
        return torch.cat(logits_list, dim=1)

    def generate_step(self, x, h_dec, enc_outputs, pad_mask=None):
        embed         = self.embedding(x.squeeze(1))
        context, attn = self.attention(enc_outputs, h_dec, mask=pad_mask)
        h_new         = self.decoder_cell(torch.cat([embed, context], dim=1), h_dec)
        return self.output_layer(h_new), h_new

    def encode_prefix(self, prefix_indices):
        return self._encode(self.embedding(prefix_indices))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    VOCAB, EMBED, HIDDEN = 35, 32, 128
    rnn  = VanillaRNN(VOCAB, EMBED, HIDDEN, dropout=0.3)
    blst = BidirectionalLSTM(VOCAB, EMBED, HIDDEN, dropout=0.3)
    attn = AttentionRNN(VOCAB, EMBED, HIDDEN, attn_size=64, dropout=0.3)
    print(f"VanillaRNN        : {rnn.count_parameters():>8,}")
    print(f"BidirectionalLSTM : {blst.count_parameters():>8,}")
    print(f"AttentionRNN      : {attn.count_parameters():>8,}")

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)

    def forward(self, x, h_prev):
        return torch.tanh(x @ self.W_ih.T + self.b_ih + h_prev @ self.W_hh.T + self.b_hh)

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.hidden_size, device=device)


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn_cell     = VanillaRNNCell(embed_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.drop         = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        batch_size, seq_len = inputs.shape
        embeds = self.drop(self.embedding(inputs))
        h = self.rnn_cell.init_hidden(batch_size, inputs.device)
        logits_list = []
        for t in range(seq_len):
            h = self.rnn_cell(embeds[:, t, :], h)
            logits_list.append(self.output_layer(self.drop(h)).unsqueeze(1))
        return torch.cat(logits_list, dim=1)

    def generate_step(self, x, h):
        embed = self.embedding(x.squeeze(1))
        h_new = self.rnn_cell(embed, h)
        return self.output_layer(h_new), h_new

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.W_all = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.U_all = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b_all = nn.Parameter(torch.zeros(4 * hidden_size))
        nn.init.xavier_uniform_(self.W_all)
        nn.init.xavier_uniform_(self.U_all)

    def forward(self, x, h_prev, c_prev):
        gates = x @ self.W_all.T + h_prev @ self.U_all.T + self.b_all
        f, i, g, o = gates.chunk(4, dim=1)
        c_t = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        h_t = torch.sigmoid(o) * torch.tanh(c_t)
        return h_t, c_t

    def init_states(self, batch_size: int, device: torch.device):
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))


class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_fwd  = LSTMCell(embed_size, hidden_size)
        self.encoder_bwd  = LSTMCell(embed_size, hidden_size)
        self.bridge_h     = nn.Linear(2 * hidden_size, hidden_size)
        self.bridge_c     = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder_cell = LSTMCell(embed_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.drop         = nn.Dropout(dropout)

    def _encode(self, embeds, lengths):
        batch_size, seq_len, _ = embeds.shape
        device = embeds.device
        hf, cf = self.encoder_fwd.init_states(batch_size, device)
        hb, cb = self.encoder_bwd.init_states(batch_size, device)
        for t in range(seq_len):
            hf, cf = self.encoder_fwd(embeds[:, t, :], hf, cf)
        for t in reversed(range(seq_len)):
            hb, cb = self.encoder_bwd(embeds[:, t, :], hb, cb)
        h_dec = torch.tanh(self.bridge_h(torch.cat([hf, hb], dim=1)))
        c_dec = torch.tanh(self.bridge_c(torch.cat([cf, cb], dim=1)))
        return h_dec, c_dec

    def forward(self, inputs, lengths):
        embeds = self.drop(self.embedding(inputs))
        h, c   = self._encode(embeds, lengths)
        logits_list = []
        for t in range(inputs.shape[1]):
            h, c = self.decoder_cell(embeds[:, t, :], h, c)
            logits_list.append(self.output_layer(self.drop(h)).unsqueeze(1))
        return torch.cat(logits_list, dim=1)

    def generate_step(self, x, h, c):
        embed = self.embedding(x.squeeze(1))
        h_new, c_new = self.decoder_cell(embed, h, c)
        return self.output_layer(h_new), h_new, c_new

    def get_initial_decoder_state(self, inputs, lengths, device):
        if inputs is None or inputs.numel() == 0:
            return (torch.zeros(1, self.hidden_size, device=device),
                    torch.zeros(1, self.hidden_size, device=device))
        return self._encode(self.embedding(inputs), lengths)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BahdanauAttention(nn.Module):
    """Additive attention: score = V_a · tanh(W_a·enc_h + U_a·dec_h)"""

    def __init__(self, hidden_size: int, attn_size: int):
        super().__init__()
        self.W_a = nn.Parameter(torch.empty(attn_size, hidden_size))
        self.U_a = nn.Parameter(torch.empty(attn_size, hidden_size))
        self.V_a = nn.Parameter(torch.empty(1, attn_size))
        nn.init.xavier_uniform_(self.W_a)
        nn.init.xavier_uniform_(self.U_a)
        nn.init.xavier_uniform_(self.V_a)

    def forward(self, enc_outputs, dec_hidden, mask=None):
        src_len      = enc_outputs.shape[1]
        W_enc        = enc_outputs @ self.W_a.T
        U_dec        = (dec_hidden @ self.U_a.T).unsqueeze(1).expand(-1, src_len, -1)
        energy       = (torch.tanh(W_enc + U_dec) @ self.V_a.T).squeeze(-1)
        if mask is not None:
            energy = energy.masked_fill(mask, float("-inf"))
        attn_weights = F.softmax(energy, dim=1)
        context      = (attn_weights.unsqueeze(1) @ enc_outputs).squeeze(1)
        return context, attn_weights


class AttentionRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int,
                 attn_size: int = 64, dropout: float = 0.0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.hidden_size  = hidden_size
        self.embedding    = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder_cell = VanillaRNNCell(embed_size, hidden_size)
        self.attention    = BahdanauAttention(hidden_size, attn_size)
        self.decoder_cell = VanillaRNNCell(embed_size + hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.drop         = nn.Dropout(dropout)

    def _encode(self, embeds):
        batch_size, seq_len, _ = embeds.shape
        h = self.encoder_cell.init_hidden(batch_size, embeds.device)
        enc_outputs = []
        for t in range(seq_len):
            h = self.encoder_cell(embeds[:, t, :], h)
            enc_outputs.append(h.unsqueeze(1))
        return torch.cat(enc_outputs, dim=1), h

    def forward(self, inputs, lengths):
        embeds   = self.drop(self.embedding(inputs))
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

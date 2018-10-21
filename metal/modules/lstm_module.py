import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTMModule(nn.Module):
    """An LSTM-based input module"""

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size=None,
        embeddings=None,
        lstm_reduction="max",
        freeze=False,
        bidirectional=True,
        verbose=True,
        **lstm_kwargs,
    ):
        """
        Args:
            embed_size: The (integer) size of the input at each time
                step; usually this is the size of the embeddings
            hidden_size: The size of the hidden layer in the LSTM
            vocab_size: The size of the vocabulary of the embeddings
                If embeddings=None, this helps to set the size of the randomly
                    initilialized embeddings
                If embeddings!=None, this is used to double check that the
                    provided embeddings have the intended size
            embeddings: An optional embedding Tensor
            lstm_reduction: One of ['mean', 'max', 'last', 'attention']
                denoting what to return as the output of the LSTMLayer
            freeze: If False, allow the embeddings to be updated
        """
        super().__init__()
        self.lstm_reduction = lstm_reduction
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.verbose = verbose

        # Load provided embeddings or randomly initialize new ones
        if embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embed_size)
            if self.verbose:
                print(f"Using randomly initialized embeddings.")
        else:
            self.embeddings = self._load_pretrained(embeddings)
            if self.verbose:
                print(f"Using pretrained embeddings.")

        # Freeze or not
        self.embeddings.weight.requires_grad = not freeze

        if self.verbose:
            print(
                f"Embeddings shape = ({self.embeddings.num_embeddings}, "
                f"{self.embeddings.embedding_dim})"
            )
            print(f"The embeddings are {'' if freeze else 'NOT '}FROZEN")
            print(f"Using lstm_reduction = '{lstm_reduction}'")

        # Create lstm core
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            **lstm_kwargs,
        )
        if lstm_reduction == "attention":
            raise NotImplementedError
            # self.attention = Attention(hidden_size *
            #   (self.lstm.bidirectional + 1))

    def _load_pretrained(self, pretrained):
        if not pretrained.dim() == 2:
            msg = (
                f"Provided embeddings have shape {pretrained.shape}. "
                "Expected a 2-dimensional tesnor."
            )
            raise ValueError(msg)
        rows, cols = pretrained.shape
        embedding = nn.Embedding(num_embeddings=rows, embedding_dim=cols)
        embedding.weight.data.copy_(pretrained)
        return embedding

    def reset_parameters(self):
        # Note: Classifier.reset() calls reset_parameters() recursively on all
        # children, so this method need not reset children modules such as
        # nn.lstm or nn.Embedding
        pass

    def _reduce_output(self, outputs, seq_lengths):
        """Reduces the output of an LSTM step

        Args:
            outputs: (torch.FloatTensor) the hidden state outputs from the
                lstm, with shape [batch_size, max_seq_length, hidden_size]
        """
        batch_size = outputs.shape[0]
        reduced = []
        # Necessary to iterate over batch because of different sequence lengths
        for i in range(batch_size):
            if self.lstm_reduction == "mean":
                # Average over all non-padding reduced
                # Use dim=0 because first dimension disappears after indexing
                reduced.append(outputs[i, : seq_lengths[i], :].mean(dim=0))
            elif self.lstm_reduction == "max":
                # Max-pool over all non-padding reduced
                # Use dim=0 because first dimension disappears after indexing
                reduced.append(outputs[i, : seq_lengths[i], :].max(dim=0)[0])
            elif self.lstm_reduction == "last":
                # Take the last output of the sequence (before padding starts)
                # NOTE: maybe better to take first and last?
                reduced.append(outputs[i, seq_lengths[i] - 1, :])
            elif self.lstm_reduction == "attention":
                raise NotImplementedError
                # reduced.append(self.attention(outputs))
            else:
                msg = (
                    f"Did not recognize lstm kwarg 'lstm_reduction' == "
                    f"{self.lstm_reduction}"
                )
                raise ValueError(msg)
        return torch.stack(reduced, dim=0)

    def forward(self, X):
        """Applies one step of an lstm (plus reduction) to the input X

        Args:
            X: (torch.LongTensor) of shape [batch_size, max_seq_length].
                The indices of the embeddings to look up for each item in the
                batch.
        """
        # Identify the first non-zero integer from the right (i.e., the length
        # of the sequence before padding starts).
        batch_size, max_seq = X.shape
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            for j in range(max_seq - 1, -1, -1):
                if X[i, j] != 0:
                    seq_lengths[i] = j + 1
                    break
        # Sort by length because pack_padded_sequence requires it
        # Save original order to restore before returning
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx, :]
        inv_perm_idx = torch.tensor(
            [i for i, _ in sorted(enumerate(perm_idx), key=lambda idx: idx[1])],
            dtype=torch.long,
        )

        X_encoded = self.embeddings(X)
        X_packed = rnn_utils.pack_padded_sequence(
            X_encoded, seq_lengths, batch_first=True
        )

        outputs, (h_t, c_t) = self.lstm(X_packed)

        outputs_unpacked, _ = rnn_utils.pad_packed_sequence(
            outputs, batch_first=True
        )
        reduced = self._reduce_output(outputs_unpacked, seq_lengths)
        return reduced[inv_perm_idx, :]


# TODO: Test this
# class Attention(nn.Module):
#     def __init__(self, attention_size):
#         """
#         An attention layer

#         Args:
#             attention_size: scalar that should match the LSTM hidden/output
#                 size or 2x that size if the lstm is bidirectional
#         """
#         super(Attention, self).__init__()
#         self.attention = nn.Parameter(torch.FloatTensor(attention_size, 1))
#         nn.init.xavier_normal(self.attention)

#     def forward(self, x_in):
#         attention_score = torch.matmul(x_in, self.attention).squeeze()
#         attention_score = F.softmax(attention_score).view(
#             x_in.size(0), x_in.size(1), 1)
#         scored_x = x_in * attention_score
#         x_out = torch.sum(scored_x, dim=1)
#         return x_out

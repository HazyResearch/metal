import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    """The Encoder implements the encode() method, which maps a batch of data to
    encoded output of dimension [batch_size, max_seq_len, encoded_size]

    The first argument must be the encoded size of the Encoder output.

    Args:
        encoded_size: (int) Output feature dimension of the Encoder
    """

    def __init__(self, encoded_size, verbose=True):
        super().__init__()
        self.encoded_size = encoded_size

    def encode(self, X):
        """
        Args:
            X: (torch.LongTensor) of shape [batch_size, max_seq_length,
            encoded_size], with all-0s vectors as padding.
        """
        assert X.shape[-1] == self.encoded_size
        return X.float()


class EmbeddingsEncoder(Encoder):
    def __init__(
        self,
        encoded_size,
        vocab_size=None,
        embeddings=None,
        freeze=False,
        verbose=True,
        seed=123,
        **kwargs,
    ):
        """
        Args:
            encoded_size: (in) Output feature dimension of the Encoder, and
                input feature dimension of the LSTM
            vocab_size: The size of the vocabulary of the embeddings
                If embeddings=None, this helps to set the size of the randomly
                    initialized embeddings
                If embeddings != None, this is used to double check that the
                    provided embeddings have the intended size
            embeddings: An optional embedding Tensor
            freeze: If False, allow the embeddings to be updated
        """
        super().__init__(encoded_size)
        self.verbose = verbose

        # Load provided embeddings or randomly initialize new ones
        if embeddings is None:

            # Note: Need to set seed here for deterministic init
            if seed is not None:
                self._set_seed(seed)
            self.embeddings = nn.Embedding(vocab_size, encoded_size)
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

    def _set_seed(self, seed):
        self.seed = seed
        if torch.cuda.is_available():
            # TODO: confirm this works for gpus without knowing gpu_id
            # torch.cuda.set_device(self.config['gpu_id'])
            torch.backends.cudnn.enabled = True
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

    def _load_pretrained(self, pretrained):
        if not pretrained.dim() == 2:
            msg = (
                f"Provided embeddings have shape {pretrained.shape}. "
                "Expected a 2-dimensional tensor."
            )
            raise ValueError(msg)
        rows, cols = pretrained.shape
        embedding = nn.Embedding(num_embeddings=rows, embedding_dim=cols)
        embedding.weight.data.copy_(pretrained)
        return embedding

    def encode(self, X):
        """
        Args:
            X: (torch.LongTensor) of shape [batch_size, max_seq_length],
            containing the indices of the embeddings to look up for each item in
            the batch, or 0 for padding.
        """
        return self.embeddings(X.long())


class CNNEncoder(nn.Module):
    def encode(self, X):
        """
        Args:
            X: (torch.LongTensor) of shape [batch_size, max_seq_length,
            encoded_size], with all-0s vectors as padding.
        """
        raise NotImplementedError()


class LSTMModule(nn.Module):
    """An LSTM-based input module"""

    def __init__(
        self,
        encoded_size,
        hidden_size,
        lstm_reduction="max",
        bidirectional=True,
        verbose=True,
        seed=123,
        lstm_num_layers=1,
        encoder_class=Encoder,
        encoder_kwargs={},
        **kwargs,
    ):
        """
        Args:
            encoder: An Encoder object with encode() method that maps from
                input sequences to [batch_size, max_seq_len, feature_dim]
            hidden_size: (int) The size of the hidden layer in the LSTM
            lstm_reduction: One of ['mean', 'max', 'last', 'attention']
                denoting what to return as the output of the LSTMLayer
            freeze: If False, allow the embeddings to be updated
            skip_embeddings: If True, directly accept X without using embeddings
        """
        super().__init__()
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.verbose = verbose

        # Initialize Encoder
        # Note constructing the Encoder here is helpful for e.g. Tuner, as then
        # all model params initialized here
        encoder_kwargs["verbose"] = self.verbose
        self.encoder = encoder_class(encoded_size, **encoder_kwargs)

        self.lstm_reduction = lstm_reduction
        if self.verbose:
            print(f"Using lstm_reduction = '{lstm_reduction}'")

        # Create lstm core
        # NOTE: We only pass explicitly-named kwargs here; can always add more!
        self.lstm = nn.LSTM(
            self.encoder.encoded_size,
            hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if lstm_reduction == "attention":
            att_size = hidden_size * (self.lstm.bidirectional + 1)
            att_param = nn.Parameter(torch.FloatTensor(att_size, 1))
            nn.init.xavier_normal_(att_param)
            self.attention_param = att_param

    def _attention(self, output):
        # output is of shape (seq_length, hidden_size)
        score = torch.matmul(output, self.attention_param).squeeze()
        score = F.softmax(score, dim=0).view(output.size(0), 1)
        scored_output = output * score
        condensed_output = torch.sum(scored_output, dim=0)
        return condensed_output

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
                reduced.append(self._attention(outputs[i, : seq_lengths[i], :]))
            else:
                msg = (
                    f"Did not recognize lstm kwarg 'lstm_reduction' == "
                    f"{self.lstm_reduction}"
                )
                raise ValueError(msg)
        return torch.stack(reduced, dim=0)

    def forward(self, X):
        """Applies one step of an lstm (plus reduction) to the input X, which
        is handled by self.encoder"""
        # Identify the first non-zero integer from the right (i.e., the length
        # of the sequence before padding starts).
        batch_size, max_seq = X.shape[0], X.shape[1]
        seq_lengths = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            for j in range(max_seq - 1, -1, -1):
                if not torch.all(X[i, j] == 0):
                    seq_lengths[i] = j + 1
                    break

        # Sort by length because pack_padded_sequence requires it
        # Save original order to restore before returning
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        X = X[perm_idx]
        inv_perm_idx = torch.tensor(
            [i for i, _ in sorted(enumerate(perm_idx), key=lambda idx: idx[1])],
            dtype=torch.long,
        )

        # Encode and pack input sequence
        X_packed = rnn_utils.pack_padded_sequence(
            self.encoder.encode(X), seq_lengths, batch_first=True
        )

        # Run LSTM
        outputs, (h_t, c_t) = self.lstm(X_packed)

        # Unpack and reduce outputs
        outputs_unpacked, _ = rnn_utils.pad_packed_sequence(
            outputs, batch_first=True
        )
        reduced = self._reduce_output(outputs_unpacked, seq_lengths)
        return reduced[inv_perm_idx, :]

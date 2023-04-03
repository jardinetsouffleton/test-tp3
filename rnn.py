from itertools import takewhile
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext.datasets import IWSLT2016

import einops
import wandb
from torchinfo import summary

# wget http://www.manythings.org/anki/fra-eng.zip
# unzip fra-eng.zip


class TranslationDataset(Dataset):
    def __init__(
            self,
            dataset: list,
            en_vocab: Vocab,
            fr_vocab: Vocab,
            en_tokenizer,
            fr_tokenizer,
        ):
        super().__init__()

        self.dataset = dataset
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer
    
    def __len__(self):
        """Return the number of examples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        """Return a sample.

        Args
        ----
            index: Index of the sample.

        Output
        ------
            en_tokens: English tokens of the sample, as a LongTensor.
            fr_tokens: French tokens of the sample, as a LongTensor.
        """
        # Get the strings
        en_sentence, fr_sentence = self.dataset[index]

        # To list of words
        # We also add the beggining-of-sentence and end-of-sentence tokens
        en_tokens = ['<bos>'] + self.en_tokenizer(en_sentence) + ['<eos>']
        fr_tokens = ['<bos>'] + self.fr_tokenizer(fr_sentence) + ['<eos>']

        # To list of tokens
        en_tokens = self.en_vocab(en_tokens)  # list[int]
        fr_tokens = self.fr_vocab(fr_tokens)

        return torch.LongTensor(en_tokens), torch.LongTensor(fr_tokens)


def yield_tokens(dataset, tokenizer, lang):
    """Tokenize the whole dataset and yield the tokens.
    """
    assert lang in ('en', 'fr')
    sentence_idx = 0 if lang == 'en' else 1

    for sentences in dataset:
        sentence = sentences[sentence_idx]
        tokens = tokenizer(sentence)
        yield tokens


def build_vocab(dataset: list, en_tokenizer, fr_tokenizer, min_freq: int):
    """Return two vocabularies, one for each language.
    """
    en_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, en_tokenizer, 'en'),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    en_vocab.set_default_index(en_vocab['<unk>'])  # Default token for unknown words

    fr_vocab = build_vocab_from_iterator(
        yield_tokens(dataset, fr_tokenizer, 'fr'),
        min_freq=min_freq,
        specials=SPECIALS,
    )
    fr_vocab.set_default_index(fr_vocab['<unk>'])

    return en_vocab, fr_vocab


def preprocess(
        dataset: list,
        en_tokenizer,
        fr_tokenizer,
        max_words: int,
    ) -> list:
    """Preprocess the dataset.
    Remove samples where at least one of the sentences are too long.
    Those samples takes too much memory.
    Also remove the pending '\n' at the end of sentences.
    """
    filtered = []

    for en_s, fr_s in dataset:
        if len(en_tokenizer(en_s)) >= max_words or len(fr_tokenizer(fr_s)) >= max_words:
            continue
        
        en_s = en_s.replace('\n', '')
        fr_s = fr_s.replace('\n', '')

        filtered.append((en_s, fr_s))

    return filtered


def build_datasets(
        max_sequence_length: int,
        min_token_freq: int,
        en_tokenizer,
        fr_tokenizer,
        train: list,
        val: list,
    ) -> tuple:
    """Build the training, validation and testing datasets.
    It takes care of the vocabulary creation.

    Args
    ----
        - max_sequence_length: Maximum number of tokens in each sequences.
            Having big sequences increases dramatically the VRAM taken during training.
        - min_token_freq: Minimum number of occurences each token must have
            to be saved in the vocabulary. Reducing this number increases
            the vocabularies's size.
        - en_tokenizer: Tokenizer for the english sentences.
        - fr_tokenizer: Tokenizer for the french sentences.
        - train and val: List containing the pairs (english, french) sentences.


    Output
    ------
        - (train_dataset, val_dataset): Tuple of the two TranslationDataset objects.
    """
    datasets = [
        preprocess(samples, en_tokenizer, fr_tokenizer, max_sequence_length)
        for samples in [train, val]
    ]

    en_vocab, fr_vocab = build_vocab(datasets[0], en_tokenizer, fr_tokenizer, min_token_freq)

    datasets = [
        TranslationDataset(samples, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer)
        for samples in datasets
    ]

    return datasets


def generate_batch(data_batch: list, src_pad_idx: int, tgt_pad_idx: int) -> tuple:
    """Add padding to the given batch so that all
    the samples are of the same size.

    Args
    ----
        data_batch: List of samples.
            Each sample is a tuple of LongTensors of varying size.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
    
    Output
    ------
        en_batch: Batch of tokens for the padded english sentences.
            Shape of [batch_size, max_en_len].
        fr_batch: Batch of tokens for the padded french sentences.
            Shape of [batch_size, max_fr_len].
    """
    en_batch, fr_batch = [], []
    for en_tokens, fr_tokens in data_batch:
        en_batch.append(en_tokens)
        fr_batch.append(fr_tokens)

    en_batch = pad_sequence(
        en_batch, padding_value=src_pad_idx, batch_first=True)
    fr_batch = pad_sequence(
        fr_batch, padding_value=tgt_pad_idx, batch_first=True)
    return en_batch, fr_batch

from torch.nn import Dropout
class RNNCell(nn.Module):
    """A single RNN layer.
    
    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        dropout: Dropout rate.

    Important note: This layer does not exactly the same thing as nn.RNNCell does.
    PyTorch implementation is only doing one simple pass over one token for each batch.
    This implementation is taking the whole sequence of each batch and provide the
    final hidden state along with the embeddings of each token in each sequence.
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            dropout: float
        ):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wih = nn.Linear(self.input_size, self.hidden_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.W_hidden = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.V = nn.Linear(self.hidden_size, self.hidden_size).to(device)
        self.dropout = Dropout(dropout).to(device)
        

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor) -> tuple:
        """Go through all the sequence in x, iteratively updating
        the hidden state h.

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state.
                Shape of [batch_size, hidden_size].

        Output
        ------
            y: Token embeddings.
                Shape of [batch_size, seq_len, hidden_size].
            h: Last hidden state.
                Shape of [batch_size, hidden_size].
        """
        y = []
        sequence_length = x.size()[1]
        for i in range(sequence_length):
            x_t = x[:, i, :]
            h = self.dropout(self.W_hidden(h) + self.Wih(x_t))
            h = self.relu(h)
            out = self.V(h)
            y.append(out)

        return torch.stack(y, dim=1), h


class RNN(nn.Module):
    """Implementation of an RNN based
    on https://pytorch.org/docs/stable/generated/torch.nn.RNN.html.

    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        num_layers: Number of layers (RNNCell or GRUCell).
        dropout: Dropout rate.
        model_type: Either 'RNN' or 'GRU', to select which model we want.
            This parameter can be removed if you decide to use the module `GRU`.
            Indeed, `GRU` should have exactly the same code as this module,
            but with `GRUCell` instead of `RNNCell`. We let the freedom for you
            to decide at which level you want to specialise the modules (either
            in `TranslationRNN` by creating a `GRU` or a `RNN`, or in `RNN`
            by creating a `GRUCell` or a `RNNCell`).
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            model_type: str,
        ):
        super().__init__()
        self.cells = []
        if model_type == "RNN":
            for layer in range(num_layers):
                if layer == 0:
                    self.cells.append(RNNCell(input_size, hidden_size, dropout))
                else:
                    self.cells.append(RNNCell(hidden_size, hidden_size, dropout))
        else:
            for layer in range(num_layers):
                if layer == 0:
                    self.cells.append(GRUCell(input_size, hidden_size, dropout))
                else:
                    self.cells.append(GRUCell(hidden_size, hidden_size, dropout))

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor=None) -> tuple:
        """Pass the input sequence through all the RNN cells.
        Returns the output and the final hidden state of each RNN layer

        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Hidden state for each RNN layer.
                Can be None, in which case an initial hidden state is created.
                Shape of [batch_size, n_layers, hidden_size].

        Output
        ------
            y: Output embeddings for each token after the RNN layers.
                Shape of [batch_size, seq_len, hidden_size].
            h: Final hidden state.
                Shape of [batch_size, n_layers, hidden_size].
        """
        if h is None:
            batch_size = x.shape[0]
            num_layers = len(self.cells)
            h = torch.zeros(batch_size, num_layers, self.cells[0].hidden_size, device=x.device)
        
        hidden_states = []
        
        for i, cell in enumerate(self.cells):
            h_i = h[:, i, :]
            x, h_i = cell(x, h_i)
            hidden_states.append(h_i)

        return x, torch.stack(hidden_states, dim=1)

class GRUCell(nn.Module):
    """A single GRU layer.
    
    Parameters
    ----------
        input_size: Size of each input token.
        hidden_size: Size of each RNN hidden state.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout).to(device)
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size).to(device)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size).to(device)

    def forward(self, x: torch.FloatTensor, h: torch.FloatTensor) -> tuple:
        """
        Args
        ----
            x: Input sequence.
                Shape of [batch_size, seq_len, input_size].
            h: Initial hidden state.
                Shape of [batch_size, hidden_size].

        Output
        ------
            y: Token embeddings.
                Shape of [batch_size, seq_len, hidden_size].
            h: Last hidden state.
                Shape of [batch_size, hidden_size].
        """
        embeddings = []
        sequence_length = x.size()[1]
        for i in range(sequence_length):
            x_t = x[:, i, :]
            x_h = torch.cat((x_t, h), dim=1)
            z = torch.sigmoid(self.update_gate(x_h))
            r = torch.sigmoid(self.reset_gate(x_h))
            h_tilde = torch.tanh(self.output_gate(torch.cat((x_t, r * h), dim=1)))
            h = (1 - z) * h_tilde + z * h
            embeddings.append(h)
        
        y = torch.stack(embeddings, dim=1)
        return y, h


class TranslationRNN(nn.Module):
    """Basic RNN encoder and decoder for a translation task.
    It can run as a vanilla RNN or a GRU-RNN.

    Parameters
    ----------
        n_tokens_src: Number of tokens in the source vocabulary.
        n_tokens_tgt: Number of tokens in the target vocabulary.
        dim_embedding: Dimension size of the word embeddings (for both language).
        dim_hidden: Dimension size of the hidden layers in the RNNs
            (for both the encoder and the decoder).
        n_layers: Number of layers in the RNNs.
        dropout: Dropout rate.
        src_pad_idx: Source padding index value.
        tgt_pad_idx: Target padding index value.
        model_type: Either 'RNN' or 'GRU', to select which model we want.
    """

    def __init__(
        self,
        n_tokens_src: int,
        n_tokens_tgt: int,
        dim_embedding: int,
        dim_hidden: int,
        n_layers: int,
        dropout: float,
        src_pad_idx: int,
        tgt_pad_idx: int,
        model_type: str,
    ):
        super().__init__()

        self.encoder_embedding = nn.Embedding(n_tokens_src, dim_embedding, padding_idx=src_pad_idx)
        self.decoder_embedding = nn.Embedding(n_tokens_tgt, dim_embedding, padding_idx=tgt_pad_idx)

        if model_type == "RNN":
            self.encoder = RNN(dim_embedding, dim_hidden, n_layers, dropout, model_type="RNN")
            self.decoder = RNN(dim_embedding, dim_hidden, n_layers, dropout, model_type="RNN")
        elif model_type == "GRU":
            self.encoder = RNN(dim_embedding, dim_hidden, n_layers, dropout, model_type="GRU")
            self.decoder = RNN(dim_embedding, dim_hidden, n_layers, dropout, model_type="GRU")
        else:
            raise ValueError("Invalid model_type. Choose either 'RNN' or 'GRU'.")

        self.output_projection = nn.Linear(dim_hidden, n_tokens_tgt)

    def forward(
        self,
        source: torch.LongTensor,
        target: torch.LongTensor
    ) -> torch.FloatTensor:
        """Predict the target tokens logites based on the source tokens.

        Args
        ----
            source: Batch of source sentences.
                Shape of [batch_size, src_seq_len].
            target: Batch of target sentences.
                Shape of [batch_size, tgt_seq_len].
        
        Output
        ------
            y: Distributions over the next token for all tokens in each sentences.
                Those need to be the logits only, do not apply a softmax because
                it will be done in the loss computation for numerical stability.
                See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for more informations.
                Shape of [batch_size, tgt_seq_len, n_tokens_tgt].
        """
        source_embedded = self.encoder_embedding(source)
        target_embedded = self.decoder_embedding(target)

        _, encoder_hidden_states = self.encoder(source_embedded)
        decoder_output, _ = self.decoder(target_embedded, encoder_hidden_states)

        logits = self.output_projection(decoder_output)

        return logits

def print_logs(dataset_type: str, logs: dict):
    """Print the logs.

    Args
    ----
        dataset_type: Either "Train", "Eval", "Test" type.
        logs: Containing the metric's name and value.
    """
    desc = [
        f'{name}: {value:.2f}'
        for name, value in logs.items()
    ]
    desc = '\t'.join(desc)
    desc = f'{dataset_type} -\t' + desc
    desc = desc.expandtabs(5)
    print(desc)


def topk_accuracy(
    real_tokens: torch.FloatTensor,
    probs_tokens: torch.FloatTensor,
    k: int,
    tgt_pad_idx: int,
) -> torch.FloatTensor:
    """Compute the top-k accuracy.
    We ignore the PAD tokens.

    Args
    ----
        real_tokens: Real tokens of the target sentence.
            Shape of [batch_size * n_tokens].
        probs_tokens: Tokens probability predicted by the model.
            Shape of [batch_size * n_tokens, n_target_vocabulary].
        k: Top-k accuracy threshold.
        src_pad_idx: Source padding index value.
    
    Output
    ------
        acc: Scalar top-k accuracy value.
    """
    total = (real_tokens != tgt_pad_idx).sum()

    _, pred_tokens = probs_tokens.topk(
        k=k, dim=-1)  # [batch_size * n_tokens, k]
    # [batch_size * n_tokens, k]
    real_tokens = einops.repeat(real_tokens, 'b -> b k', k=k)

    good = (pred_tokens == real_tokens) & (real_tokens != tgt_pad_idx)
    acc = good.sum() / total
    return acc


def indices_terminated(
    target: torch.FloatTensor,
    eos_token: int
) -> tuple:
    """Split the target sentences between the terminated and the non-terminated
    sentence. Return the indices of those two groups.

    Args
    ----
        target: The sentences.
            Shape of [batch_size, n_tokens].
        eos_token: Value of the End-of-Sentence token.

    Output
    ------
        terminated: Indices of the terminated sentences (who's got the eos_token).
            Shape of [n_terminated, ].
        non-terminated: Indices of the unfinished sentences.
            Shape of [batch_size-n_terminated, ].
    """
    terminated = [i for i, t in enumerate(target) if eos_token in t]
    non_terminated = [i for i, t in enumerate(target) if eos_token not in t]
    return torch.LongTensor(terminated), torch.LongTensor(non_terminated)


def append_beams(
    target: torch.FloatTensor,
    beams: torch.FloatTensor
) -> torch.FloatTensor:
    """Add the beam tokens to the current sentences.
    Duplicate the sentences so one token is added per beam per batch.

    Args
    ----
        target: Batch of unfinished sentences.
            Shape of [batch_size, n_tokens].
        beams: Batch of beams for each sentences.
            Shape of [batch_size, n_beams].

    Output
    ------
        target: Batch of sentences with one beam per sentence.
            Shape of [batch_size * n_beams, n_tokens+1].
    """
    batch_size, n_beams = beams.shape
    n_tokens = target.shape[1]

    # [batch_size, n_beams, n_tokens]
    target = einops.repeat(target, 'b t -> b c t', c=n_beams)
    beams = beams.unsqueeze(dim=2)  # [batch_size, n_beams, 1]

    # [batch_size, n_beams, n_tokens+1]
    target = torch.cat((target, beams), dim=2)
    # [batch_size * n_beams, n_tokens+1]
    target = target.view(batch_size*n_beams, n_tokens+1)
    return target


def beam_search(
    model: nn.Module,
    source: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    src_tokenizer,
    device: str,
    beam_width: int,
    max_target: int,
    max_sentence_length: int,
) -> list:
    """Do a beam search to produce probable translations.

    Args
    ----
        model: The translation model. Assumes it produces linear score (before softmax).
        source: The sentence to translate.
        src_vocab: The source vocabulary.
        tgt_vocab: The target vocabulary.
        device: Device to which we make the inference.
        beam_width: Number of top-k tokens we keep at each stage.
        max_target: Maximum number of target sentences we keep at the end of each stage.
        max_sentence_length: Maximum number of tokens for the translated sentence.

    Output
    ------
        sentences: List of sentences orderer by their likelihood.
    """
    src_tokens = ['<bos>'] + src_tokenizer(source) + ['<eos>']
    src_tokens = src_vocab(src_tokens)

    tgt_tokens = ['<bos>']
    tgt_tokens = tgt_vocab(tgt_tokens)

    # To tensor and add unitary batch dimension
    src_tokens = torch.LongTensor(src_tokens).to(device)
    tgt_tokens = torch.LongTensor(tgt_tokens).unsqueeze(dim=0).to(device)
    target_probs = torch.FloatTensor([1]).to(device)
    model.to(device)

    EOS_IDX = tgt_vocab['<eos>']
    with torch.no_grad():
        while tgt_tokens.shape[1] < max_sentence_length:
            batch_size, n_tokens = tgt_tokens.shape

            # Get next beams
            src = einops.repeat(src_tokens, 't -> b t', b=tgt_tokens.shape[0])
            predicted = model.forward(src, tgt_tokens)
            predicted = torch.softmax(predicted, dim=-1)
            probs, predicted = predicted[:, -1].topk(k=beam_width, dim=-1)

            # Separe between terminated sentences and the others
            idx_terminated, idx_not_terminated = indices_terminated(
                tgt_tokens, EOS_IDX)
            idx_terminated, idx_not_terminated = idx_terminated.to(
                device), idx_not_terminated.to(device)

            tgt_terminated = torch.index_select(
                tgt_tokens, dim=0, index=idx_terminated)
            tgt_probs_terminated = torch.index_select(
                target_probs, dim=0, index=idx_terminated)

            def filter_t(t): return torch.index_select(
                t, dim=0, index=idx_not_terminated)
            tgt_others = filter_t(tgt_tokens)
            tgt_probs_others = filter_t(target_probs)
            predicted = filter_t(predicted)
            probs = filter_t(probs)

            # Add the top tokens to the previous target sentences
            tgt_others = append_beams(tgt_others, predicted)

            # Add padding to terminated target
            padd = torch.zeros((len(tgt_terminated), 1),
                               dtype=torch.long, device=device)
            tgt_terminated = torch.cat(
                (tgt_terminated, padd),
                dim=1
            )

            # Update each target sentence probabilities
            tgt_probs_others = torch.repeat_interleave(
                tgt_probs_others, beam_width)
            tgt_probs_others *= probs.flatten()
            tgt_probs_terminated *= 0.999  # Penalize short sequences overtime

            # Group up the terminated and the others
            target_probs = torch.cat(
                (tgt_probs_others, tgt_probs_terminated),
                dim=0
            )
            tgt_tokens = torch.cat(
                (tgt_others, tgt_terminated),
                dim=0
            )

            # Keep only the top `max_target` target sentences
            if target_probs.shape[0] <= max_target:
                continue

            target_probs, indices = target_probs.topk(k=max_target, dim=0)
            tgt_tokens = torch.index_select(tgt_tokens, dim=0, index=indices)

    sentences = []
    for tgt_sentence in tgt_tokens:
        tgt_sentence = list(tgt_sentence)[1:]  # Remove <bos> token
        tgt_sentence = list(takewhile(lambda t: t != EOS_IDX, tgt_sentence))
        tgt_sentence = ' '.join(tgt_vocab.lookup_tokens(tgt_sentence))
        sentences.append(tgt_sentence)

    sentences = [beautify(s) for s in sentences]

    # Join the sentences with their likelihood
    sentences = [(s, p.item()) for s, p in zip(sentences, target_probs)]
    # Sort the sentences by their likelihood
    sentences = [(s, p) for s, p in sorted(sentences, key=lambda k: k[1], reverse=True)]

    return sentences


def beautify(sentence: str) -> str:
    """Removes useless spaces.
    """
    punc = {'.', ',', ';'}
    for p in punc:
        sentence = sentence.replace(f' {p}', p)

    links = {'-', "'"}
    for l in links:
        sentence = sentence.replace(f'{l} ', l)
        sentence = sentence.replace(f' {l}', l)

    return sentence

def loss_batch(
    model: nn.Module,
    source: torch.LongTensor,
    target: torch.LongTensor,
    config: dict,
) -> dict:
    """Compute the metrics associated with this batch.
    The metrics are:
        - loss
        - top-1 accuracy
        - top-5 accuracy
        - top-10 accuracy

    Args
    ----
        model: The model to train.
        source: Batch of source tokens.
            Shape of [batch_size, n_src_tokens].
        target: Batch of target tokens.
            Shape of [batch_size, n_tgt_tokens].
        config: Additional parameters.

    Output
    ------
        metrics: Dictionnary containing evaluated metrics on this batch.
    """
    device = config['device']
    loss_fn = config['loss'].to(device)
    metrics = dict()

    source, target = source.to(device), target.to(device)
    target_in, target_out = target[:, :-1], target[:, 1:]

    # Loss
    pred = model(source, target_in)  # [batch_size, n_tgt_tokens-1, n_vocab]
    # [batch_size * (n_tgt_tokens - 1), n_vocab]
    pred = pred.view(-1, pred.shape[2])
    target_out = target_out.flatten()  # [batch_size * (n_tgt_tokens - 1),]
    metrics['loss'] = loss_fn(pred, target_out)

    # Accuracy - we ignore the padding predictions
    for k in [1, 5, 10]:
        metrics[f'top-{k}'] = topk_accuracy(target_out,
                                            pred, k, config['tgt_pad_idx'])

    return metrics


def eval_model(model: nn.Module, dataloader: DataLoader, config: dict) -> dict:
    """Evaluate the model on the given dataloader.
    """
    device = config['device']
    logs = defaultdict(list)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for source, target in dataloader:
            metrics = loss_batch(model, source, target, config)
            for name, value in metrics.items():
                logs[name].append(value.cpu().item())

    for name, values in logs.items():
        logs[name] = np.mean(values)
    return logs


def train_model(model: nn.Module, config: dict):
    """Train the model in a teacher forcing manner.
    """
    train_loader, val_loader = config['train_loader'], config['val_loader']
    train_dataset, val_dataset = train_loader.dataset.dataset, val_loader.dataset.dataset
    optimizer = config['optimizer']
    clip = config['clip']
    device = config['device']

    columns = ['epoch']
    for mode in ['train', 'validation']:
        columns += [
            f'{mode} - {colname}'
            for colname in ['source', 'target', 'predicted', 'likelihood']
        ]
    log_table = wandb.Table(columns=columns)

    print(f'Starting training for {config["epochs"]} epochs, using {device}.')
    for e in range(config['epochs']):
        print(f'\nEpoch {e+1}')

        model.to(device)
        model.train()
        logs = defaultdict(list)

        for batch_id, (source, target) in enumerate(train_loader):
            optimizer.zero_grad()

            metrics = loss_batch(model, source, target, config)
            loss = metrics['loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            for name, value in metrics.items():
                # Don't forget the '.item' to free the cuda memory
                logs[name].append(value.cpu().item())

            if batch_id % config['log_every'] == 0:
                for name, value in logs.items():
                    logs[name] = np.mean(value)

                train_logs = {
                    f'Train - {m}': v
                    for m, v in logs.items()
                }
                wandb.log(train_logs)
                logs = defaultdict(list)

        # Logs
        if len(logs) != 0:
            for name, value in logs.items():
                logs[name] = np.mean(value)
            train_logs = {
                f'Train - {m}': v
                for m, v in logs.items()
            }
        else:
            logs = {
                m.split(' - ')[1]: v
                for m, v in train_logs.items()
            }

        print_logs('Train', logs)

        logs = eval_model(model, val_loader, config)
        print_logs('Eval', logs)
        val_logs = {
            f'Validation - {m}': v
            for m, v in logs.items()
        }

        val_source, val_target = val_dataset[torch.randint(
            len(val_dataset), (1,))]
        val_pred, val_prob = beam_search(
            model,
            val_source,
            config['src_vocab'],
            config['tgt_vocab'],
            config['src_tokenizer'],
            device,  # It can take a lot of VRAM
            beam_width=10,
            max_target=100,
            max_sentence_length=config['max_sequence_length'],
        )[0]
        print(val_source)
        print(val_pred)

        logs = {**train_logs, **val_logs}  # Merge dictionnaries
        wandb.log(logs)  # Upload to the WandB cloud

        # Table logs
        train_source, train_target = train_dataset[torch.randint(len(train_dataset), (1,))]
        train_pred, train_prob = beam_search(
            model,
            train_source,
            config['src_vocab'],
            config['tgt_vocab'],
            config['src_tokenizer'],
            device,  # It can take a lot of VRAM
            beam_width=10,
            max_target=100,
            max_sentence_length=config['max_sequence_length'],
        )[0]

        data = [
            e + 1,
            train_source, train_target, train_pred, train_prob,
            val_source, val_target, val_pred, val_prob,
        ]
        log_table.add_data(*data)

    # Log the table at the end of the training
    wandb.log({'Model predictions': log_table})

if __name__ == "__main__":
    import random
    # num_samples = 1000
    df = pd.read_csv('fra.txt', sep='\t', names=['english', 'french', 'attribution'])#[:num_samples]
    train = [(en, fr) for en, fr in zip(df['english'], df['french'])]
    train, valid = train_test_split(train, test_size=0.1, random_state=0)
    print(len(train))

    en_tokenizer, fr_tokenizer = get_tokenizer('spacy', language='en'), get_tokenizer('spacy', language='fr')

    SPECIALS = ['<unk>', '<pad>', '<bos>', '<eos>']
    MAX_SEQ_LEN = 60
    MIN_TOK_FREQ = 2
    train_dataset, val_dataset = build_datasets(
        MAX_SEQ_LEN,
        MIN_TOK_FREQ,
        en_tokenizer,
        fr_tokenizer,
        train,
        valid,
    )

    print(f'English vocabulary size: {len(train_dataset.en_vocab):,}')
    print(f'French vocabulary size: {len(train_dataset.fr_vocab):,}')

    print(f'\nTraining examples: {len(train_dataset):,}')
    print(f'Validation examples: {len(val_dataset):,}')

    # Build the model, the dataloaders, optimizer and the loss function
    # Log every hyperparameters and arguments into the config dictionnary
    def generate_random_search_parameters(n, batch_sizes, hidden_units, learning_rates, dim_embeddings, num_layers, dropout, model_type):
        tested_combinations = set()
        count = 0

        while count < n:
            config = {
            # General parameters
            'epochs': 20,
            'batch_size': random.choice(batch_sizes),
                'lr': random.choice(learning_rates),
            'betas': (0.9, 0.99),
            'clip': 5,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',

            # Model parameters
            'n_tokens_src': len(train_dataset.en_vocab),
            'n_tokens_tgt': len(train_dataset.fr_vocab),
            'n_heads': 4,
            'dim_embedding': random.choice(dim_embeddings),
            'dim_hidden': random.choice(hidden_units),
            'n_layers': random.choice(num_layers),
            'dropout': random.choice(dropout),
            'model_type': model_type,

            # Others
            'max_sequence_length': MAX_SEQ_LEN,
            'min_token_freq': MIN_TOK_FREQ,
            'src_vocab': train_dataset.en_vocab,
            'tgt_vocab': train_dataset.fr_vocab,
            'src_tokenizer': en_tokenizer,
            'tgt_tokenizer': fr_tokenizer,
            'src_pad_idx': train_dataset.en_vocab['<pad>'],
            'tgt_pad_idx': train_dataset.fr_vocab['<pad>'],
            'seed': 0,
            'log_every': 50,  # Number of batches between each wandb logs
        }

            frozen_params = frozenset(config.items())

            if frozen_params not in tested_combinations:
                tested_combinations.add(frozen_params)
                count += 1
                yield config
    
    
    for model_type in ["GRU", "RNN"]:
        num_runs = 15
        configs = generate_random_search_parameters(
            num_runs,
            batch_sizes=[64, 128], 
            hidden_units=[128, 256, 512], 
            learning_rates=[0.001, 0.0015],
            dim_embeddings=[256, 400, 512],
            num_layers=[2, 3, 4],
            dropout=[0.1, 0.2],
            model_type=model_type
        )
            
        for config in configs:
            torch.manual_seed(config['seed'])

            config['train_loader'] = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                collate_fn=lambda batch: generate_batch(batch, config['src_pad_idx'], config['tgt_pad_idx'])
            )

            config['val_loader'] = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                collate_fn=lambda batch: generate_batch(batch, config['src_pad_idx'], config['tgt_pad_idx'])
            )

            model = TranslationRNN(
                config['n_tokens_src'],
                config['n_tokens_tgt'],
                config['dim_embedding'],
                config['dim_hidden'],
                config['n_layers'],
                config['dropout'],
                config['src_pad_idx'],
                config['tgt_pad_idx'],
                config['model_type'],
            )
            """
            model = TranslationTransformer(
                config['n_tokens_src'],
                config['n_tokens_tgt'],
                config['n_heads'],
                config['dim_embedding'],
                config['dim_hidden'],
                config['n_layers'],
                config['dropout'],
                config['src_pad_idx'],
                config['tgt_pad_idx'],
            )
            """

            config['optimizer'] = optim.Adam(
                model.parameters(),
                lr=config['lr'],
                betas=config['betas'],
            )

            weight_classes = torch.ones(config['n_tokens_tgt'], dtype=torch.float)
            weight_classes[config['tgt_vocab']['<unk>']] = 0.1  # Lower the importance of that class
            config['loss'] = nn.CrossEntropyLoss(
                weight=weight_classes,
                ignore_index=config['tgt_pad_idx'],  # We do not have to learn those
            )

            summary(
                model,
                input_size=[
                    (config['batch_size'], config['max_sequence_length']),
                    (config['batch_size'], config['max_sequence_length'])
                ],
                dtypes=[torch.long, torch.long],
                depth=3,
            )
            with wandb.init(
                config=config,
                project='INF8225 - TP3',  # Title of your project
                group=config["model_type"],  # In what group of runs do you want this run to be in?
                name=f'bs={config["batch_size"]}-hi={config["dim_hidden"]}-emb={config["dim_embeddings"]}-l={config["n_layers"]}-lr={config["lr"]}-dr={config["dropout"]}',
                save_code=True,
            ):
                train_model(model, config)

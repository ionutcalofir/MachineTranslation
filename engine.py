import time
import math
import torch.nn as nn
import torch.utils.data
from torchtext.data.metrics import bleu_score

from dataset import EOS_IDX, PAD_IDX, SOS_IDX, UNK_IDX, MAX_TOKENS
from dataset import BaseDataset, Dataset, MyCollator, Processor
import model_transformer

class Engine:
    def __init__(self,
                 data_path,
                 name_suffix,
                 model_path,
                 batch_size=32,
                 hid_dim=256,
                 enc_layers=3,
                 dec_layers=3,
                 enc_heads=8,
                 dec_heads=8,
                 enc_pf_dim=512,
                 dec_pf_dim=512,
                 enc_dropout=0.1,
                 dec_dropout=0.1,
                 learning_rate=0.0005,
                 n_epochs=10,
                 clip=1):
        self._n_epochs = n_epochs
        self._clip = clip
        self._batch_size = batch_size
        self._model_path = model_path

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Detect if GPU is available
        self._dataset = BaseDataset(data_path, name_suffix)
        self._my_collator = MyCollator()
        self._processor = Processor()

        self._data_loader_train = \
            torch.utils.data.DataLoader(Dataset(data=self._dataset.train,
                                                vocab_l0=self._dataset.vocab_l0,
                                                vocab_l1=self._dataset.vocab_l1,
                                                phase='train'),
                                        batch_size=32,
                                        shuffle=False,
                                        collate_fn=self._my_collator)

        self._data_loader_val = \
            torch.utils.data.DataLoader(Dataset(data=self._dataset.val,
                                                vocab_l0=self._dataset.vocab_l0,
                                                vocab_l1=self._dataset.vocab_l1,
                                                phase='val'),
                                        batch_size=32,
                                        shuffle=False,
                                        collate_fn=self._my_collator)

        self._data_loader_test = \
            torch.utils.data.DataLoader(Dataset(data=self._dataset.test,
                                                vocab_l0=self._dataset.vocab_l0,
                                                vocab_l1=self._dataset.vocab_l1,
                                                phase='test'),
                                        batch_size=32,
                                        shuffle=False,
                                        collate_fn=self._my_collator)

        input_dim = len(self._dataset.vocab_l0)
        output_dim = len(self._dataset.vocab_l1)

        enc = model_transformer.Encoder(input_dim,
                                        hid_dim,
                                        enc_layers,
                                        enc_heads,
                                        enc_pf_dim,
                                        enc_dropout,
                                        self._device,
                                        max_length=MAX_TOKENS + 2)

        dec = model_transformer.Decoder(output_dim,
                                        hid_dim,
                                        dec_layers,
                                        dec_heads,
                                        dec_pf_dim,
                                        dec_dropout,
                                        self._device,
                                        max_length=MAX_TOKENS + 2)

        self._model = model_transformer.Seq2Seq(enc, dec, PAD_IDX, PAD_IDX, self._device).to(self._device)
        self._count_parameters(self._model)
        self._model.apply(self._initialize_weights)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def _count_parameters(self, model):
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The model has {} trainable parameters'.format(parameters))

    def _initialize_weights(self, m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _train_step_transformer(self):
        self._model.train()
        epoch_loss = 0

        for i, (src, trg) in enumerate(self._data_loader_train):
            if (i % 1000 == 0):
                print('Batch: {}/{}'.format(i, len(self._data_loader_train)))

            src = src.to(self._device)
            trg = trg.to(self._device)

            self._optimizer.zero_grad()
            output, _ = self._model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = self._criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip)
            self._optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(self._data_loader_train)

    def _evaluate_step_transformer(self, iterator):
        self._model.eval()
        epoch_loss = 0

        for i, (src, trg) in enumerate(iterator):
            if (i % 10 == 0):
                print('Batch: {}/{}'.format(i, len(iterator)))

            src = src.to(self._device)
            trg = trg.to(self._device)

            output, _ = self._model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = self._criterion(output, trg)
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def train(self):
        best_valid_loss = float('inf')
        for epoch in range(self._n_epochs):
            start_time = time.time()
            train_loss = self._train_step_transformer()
            valid_loss = self._evaluate_step_transformer(self._data_loader_val)
            end_time = time.time()

            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self._model.state_dict(), self._model_path)

            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    def test(self, model_path=None):
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))

        test_loss = self._evaluate_step_transformer(self._data_loader_test)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

        bleu_score = self.calculate_bleu(model_path=None)
        print('Bleu score: {}'.format(bleu_score * 100.))

    def translate_sentence(self, sentence, model_path=None):
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._model.eval()

        tokens = self._processor.tokenize(sentence)
        tokens_ids = self._processor.tokens2ids(tokens, self._dataset.vocab_l0)
        tokens_ids.insert(0, SOS_IDX)
        tokens_ids.append(EOS_IDX)

        src_tensor = torch.LongTensor(tokens_ids).unsqueeze(0).to(self._device)
        src_mask = self._model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = self._model.encoder(src_tensor, src_mask)

        trg_indexes = [SOS_IDX]

        for i in range(MAX_TOKENS):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self._device)
            trg_mask = self._model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                output, attention = self._model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)

            if pred_token == EOS_IDX:
                break

        inv_vocab = {val[1]: key for key, val in self._dataset.vocab_l1.items()}
        trg_tokens = [inv_vocab[i] for i in trg_indexes]
        return trg_tokens[1:], attention

    def calculate_bleu(self, model_path=None):
        trgs = []
        pred_trgs = []

        for i, (src_sent, trg_sent) in enumerate(self._dataset.test):
            if (i % 100 == 0):
                print('Test sentence: {}/{}'.format(i, len(self._dataset.test)))
            trg = self._processor.tokenize(trg_sent)

            pred_trg, _ = self.translate_sentence(src_sent, model_path=model_path)

            # cut off <eos> token
            pred_trg = pred_trg[:-1]

            pred_trgs.append(pred_trg)
            trgs.append([trg])

            # print()
            # print('English: {}'.format(src_sent))
            # print('Romanian: {}'.format(' '.join(trg)))
            # print('Predicted Romanian: {}'.format(' '.join(pred_trg)))
            # print()

        return bleu_score(pred_trgs, trgs)

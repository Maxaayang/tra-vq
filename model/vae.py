import torch
import torch.utils.data
import torch.nn as nn
import random

lstm_1_hidden_size = 88
lstm_1_layers = 2
latent_dimension = 512

lstm_conductor_hidden_size = 1024
lstm_conductor_input_size = 1     # conductor gets only zeros as inputs anyway, so just set this very small.

lstm_l2_decoder_hidden_size = 1024


class VAE(nn.Module):

    def __init__(self, bars, enc_d_model, d_vae_latent, dec_d_model, d_seg_emb, pianoroll=False):
        super(VAE, self).__init__()

        if self.training:
            self.counter = 0
            self.scheduled_sampling_ratio = 0
            self.ground_truth = None

        self.enc_d_model = enc_d_model
        self.d_vae_latent = d_vae_latent
        # self.input_size = 89 if pianoroll else 90
        self.input_size = enc_d_model
        self.pianoroll = pianoroll
        self.batch_size = 1     # gets overwritten in forward, but is used when any function except forward() is called, e.g. sample()
        self.resolution_per_beat = 4     # quantized to sixteenth notes if set to 4 (there are 4 sixteenth notes per beat)
        self.seq_len = bars * 4 * self.resolution_per_beat
        self.u = bars       # amount of subsequences that conductor layer creates

        # encoder
        self.lstm_1 = nn.LSTM(input_size=self.enc_d_model, hidden_size=lstm_1_hidden_size, num_layers=lstm_1_layers, bidirectional=True, batch_first=True)
        self.fc_mean = nn.Linear(in_features=128, out_features=d_vae_latent)
        self.fc_std_deviation = nn.Linear(in_features=128, out_features=d_vae_latent)

        # decoder
            #conductor

        self.seg_emb_proj = nn.Linear(d_seg_emb, dec_d_model, bias=False)
        self.fc_2 = nn.Linear(in_features=dec_d_model, out_features=lstm_conductor_hidden_size*4)      # output is used to initialize h and c for both layers of the conductor lstm
        self.lstm_conductor = nn.LSTM(input_size=lstm_conductor_input_size, hidden_size=lstm_conductor_hidden_size, num_layers=2, batch_first=True)

            # second level decoder

        self.fc_3 = nn.Linear(in_features=lstm_conductor_hidden_size, out_features=lstm_l2_decoder_hidden_size*4)   # output is used to initialize h and c for both layers of the l2 lstm
        self.lstm_l2_decoder_cell_1 = nn.LSTMCell(input_size=lstm_conductor_hidden_size+self.input_size, hidden_size=lstm_l2_decoder_hidden_size)
        self.lstm_l2_decoder_cell_2 = nn.LSTMCell(input_size=lstm_l2_decoder_hidden_size, hidden_size=lstm_l2_decoder_hidden_size)
        self.fc_4 = nn.Linear(in_features=lstm_l2_decoder_hidden_size, out_features=self.enc_d_model)


    def encode(self, t):
        # 128, 176, 512
        self.batch_size = t.shape[0]

        # input of shape (batch, seq_len, input_size)
        # hidden of shape  (num_layers * num_directions, batch, hidden_size) only if batch_first == False!!

        _, (h, _) = self.lstm_1(t)  # 4, 128, 2048

        h_t = h.view(lstm_1_layers, 2, t.shape[1], 64)    # 2 = num_directions
        h_t_forward = h_t[1, 0, :, :]  # 128, 2048
        h_t_backward = h_t[1, 1, :, :]  # 128, 2048
        h_t = torch.cat((h_t_forward, h_t_backward), dim=1)    # 128, 4096

        # h_t = h_t
        # h_t = h_t.view(t.shape[1], t.shape[2])  # 176, 512

        z_mean = self.fc_mean(h_t) # 176, 128

        z_std_deviation = self.fc_std_deviation(h_t)
        z_std_deviation = torch.log1p(torch.exp(z_std_deviation))

        return z_mean, z_std_deviation


    def reparameterize(self, mean, std_deviation):
        return mean + torch.randn_like(mean) * std_deviation


    def l2_decode(self, embedding, previous):
        t = self.fc_3(embedding)
        t = torch.tanh(t)

        h1 = t[:, 0:lstm_l2_decoder_hidden_size]
        h2 = t[:, lstm_l2_decoder_hidden_size:2 * lstm_l2_decoder_hidden_size]
        c1 = t[:, 2 * lstm_l2_decoder_hidden_size:3 * lstm_l2_decoder_hidden_size]
        c2 = t[:, 3 * lstm_l2_decoder_hidden_size:4 * lstm_l2_decoder_hidden_size]

        outputs = []

        for _ in range(self.seq_len//self.u):
            if self.training:
                if self.counter > 0 and random.random() > self.scheduled_sampling_ratio:
                    previous = self.ground_truth[self.counter - 1]
                else:
                    previous = previous.detach()        # needed?

            l2_in = torch.cat((embedding, previous), dim=1)
            h1, c1 = self.lstm_l2_decoder_cell_1(l2_in, (h1, c1))
            h2, c2 = self.lstm_l2_decoder_cell_2(h1, (h2, c2))
            previous = self.fc_4(h2)
            outputs.append(previous)

        return outputs


    def decode(self, z, seg_emb):
        device = z.device

        # get initial states for conductor lstm

        seg_emb = self.seg_emb_proj(seg_emb)
        z = torch.cat((z, seg_emb), dim=1)

        t = self.fc_2(z)
        t = torch.tanh(t)

        h1 = t[:, :, 0:lstm_conductor_hidden_size]
        h2 = t[:, :, lstm_conductor_hidden_size:2 * lstm_conductor_hidden_size]
        c1 = t[:, :, 2 * lstm_conductor_hidden_size:3 * lstm_conductor_hidden_size]
        c2 = t[:, :, 3 * lstm_conductor_hidden_size:4 * lstm_conductor_hidden_size]

        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)

        # get embeddings from conductor

        conductor_input = torch.zeros(size=(self.batch_size, self.u, lstm_conductor_input_size), device=device)
        print(conductor_input.size())

        embeddings, _ = self.lstm_conductor(conductor_input, (h, c))
        embeddings = torch.unbind(embeddings, dim=1)

        # decode embeddings

        outputs = []
        previous = torch.zeros((self.batch_size, self.input_size), device=device)

        for emb in embeddings:
            l2_out = self.l2_decode(emb, previous)
            outputs.extend(l2_out)
            previous = l2_out[-1]

        output_tensor = torch.stack(outputs, dim=1)

        output_tensor = output_tensor.softmax(dim=2)

        return output_tensor


    def forward(self, t):
        self.batch_size = t.shape[0]
        z_mean, z_std_deviation = self.encode(t)
        z = self.reparameterize(z_mean, z_std_deviation)
        out = self.decode(z)
        if self.training:
            self.ground_truth = None        # not necessary, but ensures that the old ground truth cant accidentally be reused in the next step
            self.counter = 0
        return out, z_mean, z_std_deviation


    def sample(self, z=None):   # always returns a pianoroll representation
        if z is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            z = torch.randn((1, latent_dimension), requires_grad=False, device=device)


        sample = self.decode(z)
        sample = data.model_output_to_pianoroll(sample, self.pianoroll)

        return sample


    def set_ground_truth(self, ground_truth):
        self.ground_truth = ground_truth


    def set_scheduled_sampling_ratio(self, ratio):      # ratio is the probability with which the model uses its own previous output instead of teacher forcing
        raise Warning("Using scheduled sampling leads to decreased performance. You should not use it!")
        self.scheduled_sampling_ratio = ratio

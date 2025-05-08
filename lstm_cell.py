class BasicLSTM_cell(object):
    def __init__(self, input_units, hidden_units, output_units, batch_size, seq_length):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        # Weights for input gate
        self.Wi = tf.Variable(tf.zeros([input_units, hidden_units]), trainable=True)
        self.Ui = tf.Variable(tf.zeros([hidden_units, hidden_units]), trainable=True)
        self.bi = tf.Variable(tf.zeros([hidden_units]), trainable=True)

        # Forget gate
        self.Wf = tf.Variable(tf.zeros([input_units, hidden_units]), trainable=True)
        self.Uf = tf.Variable(tf.zeros([hidden_units, hidden_units]), trainable=True)
        self.bf = tf.Variable(tf.zeros([hidden_units]), trainable=True)

        # Output gate
        self.Woutg = tf.Variable(tf.zeros([input_units, hidden_units]), trainable=True)
        self.Uoutg = tf.Variable(tf.zeros([hidden_units, hidden_units]), trainable=True)
        self.boutg = tf.Variable(tf.zeros([hidden_units]), trainable=True)

        # Cell state
        self.Wc = tf.Variable(tf.zeros([input_units, hidden_units]), trainable=True)
        self.Uc = tf.Variable(tf.zeros([hidden_units, hidden_units]), trainable=True)
        self.bc = tf.Variable(tf.zeros([hidden_units]), trainable=True)

        # Output layer
        self.Wo = tf.Variable(tf.random.truncated_normal([hidden_units, output_units], mean=0, stddev=.02), trainable=True)
        self.bo = tf.Variable(tf.random.truncated_normal([output_units], mean=0, stddev=.02), trainable=True)

        # Placeholder input shape setup (just for initial hidden)
        self._inputs = tf.zeros([batch_size, seq_length, input_units])  # replace later with real input

        # Prepare input for tf.scan
        batch_input_ = tf.transpose(self._inputs, perm=[1, 0, 2])  # (seq, batch, features)
        self.processed_input = batch_input_

        # Initial hidden and cell states
        self.initial_hidden = tf.zeros([batch_size, hidden_units])
        self.initial_hidden = tf.stack([self.initial_hidden, self.initial_hidden])  # [h, c]

    def Lstm(self, previous_hidden_memory, x):
        h_prev, c_prev = tf.unstack(previous_hidden_memory)

        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(h_prev, self.Ui) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(h_prev, self.Uf) + self.bf)
        o = tf.sigmoid(tf.matmul(x, self.Woutg) + tf.matmul(h_prev, self.Uoutg) + self.boutg)
        c_ = tf.tanh(tf.matmul(x, self.Wc) + tf.matmul(h_prev, self.Uc) + self.bc)

        c = f * c_prev + i * c_
        h = o * tf.tanh(c)

        return tf.stack([h, c])

    def get_states(self):
        all_hidden_states = tf.scan(self.Lstm, self.processed_input, initializer=self.initial_hidden)
        all_hidden_states = all_hidden_states[:, 0, :, :]  # extract h only
        return all_hidden_states

    def get_output(self, hidden_state):
        return tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)

    def get_outputs(self):
        all_hidden_states = self.get_states()
        return tf.map_fn(self.get_output, all_hidden_states)

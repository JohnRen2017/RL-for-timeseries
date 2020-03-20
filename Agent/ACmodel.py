import tensorflow as tf


class Actor(tf.keras.Model):
    def __init__(
        self,
        action_dim,
        filters_C,
        kernel_size,
        hidden_R,
        dropout,
        dropout_r,
        activation,
        Hstep,
    ):
        super().__init__(name="actor")
        self.Hstep = Hstep
        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters_C,
            kernel_size=kernel_size,
            activation=activation,
            name="actor_conv1",
        )
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=2, name="actor_maxpool")
        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters_C,
            kernel_size=kernel_size,
            activation=activation,
            name="actor_conv2",
        )
        self.lstm = tf.keras.layers.LSTM(
            hidden_R, dropout=dropout, recurrent_dropout=dropout_r, name="actor_lstm"
        )

        self.permute = tf.keras.layers.Permute((2, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_R)
        self.dense2 = tf.keras.layers.Dense(hidden_R)
        self.drop = tf.keras.layers.Dropout(0.3)

        self.out = tf.keras.layers.Dense(action_dim, name="actor_activation")

    @tf.function(input_signature=(tf.TensorSpec([None, 168, 3], dtype=tf.float32),))
    def call(self, inputs, training=True):
        conv1_out = self.conv1(inputs)
        pool_out = self.maxpool(conv1_out)
        conv2_out = self.conv2(pool_out)
        lstm_out = self.lstm(conv2_out)

        x1 = self.permute(inputs[:, -self.Hstep :, -3:])
        x1 = self.flatten(x1)
        x1 = self.dense1(x1)
        x1 = self.dense2(x1)
        if training:
            out1 = self.drop(x1)

        return self.out(tf.concat([lstm_out, out1], axis=-1))


class Critic(tf.keras.Model):
    def __init__(
        self, filters_C, kernel_size, hidden_R, dropout, dropout_r, activation, Hstep
    ):
        super().__init__(name="critic")
        self.Hstep = Hstep
        self.conv1 = tf.keras.layers.Conv1D(
            filters=filters_C,
            kernel_size=kernel_size,
            activation=activation,
            name="critic_conv1",
        )
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=2, name="critic_maxpool")
        self.conv2 = tf.keras.layers.Conv1D(
            filters=filters_C,
            kernel_size=kernel_size,
            activation=activation,
            name="critic_conv2",
        )
        self.lstm = tf.keras.layers.LSTM(
            hidden_R, dropout=dropout, recurrent_dropout=dropout_r, name="critic_lstm"
        )

        self.permute = tf.keras.layers.Permute((2, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(hidden_R)
        self.dense2 = tf.keras.layers.Dense(hidden_R)
        self.drop = tf.keras.layers.Dropout(0.3)

        self.out = tf.keras.layers.Dense(1, name="critic_activation")

    @tf.function(input_signature=(tf.TensorSpec([None, 168, 3], dtype=tf.float32),))
    def call(self, inputs, training=True):
        conv1_out = self.conv1(inputs)
        pool_out = self.maxpool(conv1_out)
        conv2_out = self.conv2(pool_out)
        lstm_out = self.lstm(conv2_out)

        x1 = self.permute(inputs[:, -self.Hstep :, -3:])
        x1 = self.flatten(x1)
        x1 = self.dense1(x1)
        x1 = self.dense2(x1)
        if training:
            out1 = self.drop(x1)

        return self.out(tf.concat([lstm_out, out1], axis=-1))

import tensorflow as tf


class GRUTrainingModel(tf.keras.layers.Layer):
    def __init__(self):
        super(GRUTrainingModel, self).__init__()

        self._output_size = 512
        self._gru = tf.keras.layers.GRUCell(self._output_size)
        kernel_initializer = tf.keras.initializers.GlorotUniform()

        self.visual_head = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 8, 4, kernel_initializer=kernel_initializer, activation="relu"  # type: ignore
                ),
                tf.keras.layers.Conv2D(
                    64, 4, 2, kernel_initializer=kernel_initializer, activation="relu"  # type: ignore
                ),
                tf.keras.layers.Conv2D(
                    64, 3, 1, kernel_initializer=kernel_initializer, activation="relu"  # type: ignore
                ),
                tf.keras.layers.Flatten(),
            ]
        )
        self.vector_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    128, kernel_initializer=kernel_initializer, activation="relu"  # type: ignore
                ),
            ]
        )

        self.gru = tf.keras.layers.GRUCell(512)

        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        # self.dense3 = tf.keras.layers.Dense(512, activation='relu')

    def call(self, inputs, *args, **kwargs):
        visual_obs = inputs["visual_obs"]
        vector_obs = inputs["vector_obs"]
        # print(visual_obs.shape)
        # print(vector_obs.shape)
        visual_obs = self.visual_head(visual_obs)
        vector_obs = self.vector_head(vector_obs)

        h = tf.concat([visual_obs, vector_obs], axis=1)
        shape = tf.shape(h)

        # Remove batch dimension from tensor h
        h = tf.reshape(h, tf.math.reduce_prod(shape))

        h = self.gru(h)

        h = tf.reshape(h, shape)

        value = self.dense1(h)
        policy = self.dense2(h)

        return policy, value

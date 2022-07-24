import tensorflow as tf
import tensorflow_recommenders as tfrs


class StackingLayer(tf.keras.layers.Layer):

    def __init__(self, input_layers, output_layer):
        super(StackingLayer, self).__init__()
        self.input_layers = input_layers
        self.output_layer = output_layer

    def call(self, inputs, training):
        bottleneck_outputs = []
        for input_layer in self.input_layers:
            bottleneck_outputs.append(input_layer(inputs))
        outputs = self.output_layer(
            tf.concat(bottleneck_outputs, axis=-1)
        )
        outputs = tf.nn.l2_normalize(outputs, axis=-1)
        return outputs


class MultiTaskLoss(tf.keras.layers.Layer):

    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def build(self, input_shape):
        self.log_vars = tf.Variable(
            initial_value=tf.zeros(input_shape),
            trainable=True
        )

    def call(self, inputs):
        loss = tf.math.reduce_sum(
            tf.math.exp(-self.log_vars) * inputs + self.log_vars
        )
        return loss


class GravityRegularizationLayer(tf.keras.layers.Layer):

    def __init__(self, gamma):
        super(GravityRegularizationLayer, self).__init__()
        self.gamma = gamma

    def call(self, inputs):
        user_embedding, item_embedding = inputs
        gravity_reg = tf.math.divide(
            tf.reduce_sum(
                tf.matmul(user_embedding, user_embedding, transpose_a=True) *
                tf.matmul(item_embedding, item_embedding, transpose_a=True)
            ),
            tf.cast(
                tf.shape(user_embedding)[0] * tf.shape(item_embedding)[0],
                tf.float32
            )
        )
        return self.gamma * gravity_reg


class MultiTaskDSSM(tfrs.models.Model):

    def __init__(
        self,
        user_model,
        item_specific_model,
        item_general_model,
        specific_task_layer,
        general_task_layer,
        task_weighting_layer,
        regularizaiton_layer
    ):
        super().__init__()
        self.user_model = user_model
        self.item_specific_model = item_specific_model
        self.item_general_model = item_general_model
        self.specific_task_layer = specific_task_layer
        self.general_task_layer = general_task_layer
        self.task_weighting_layer = task_weighting_layer
        self.regularizaiton_layer = regularizaiton_layer

    def call(self, features):
        user_embedding = self.user_model(features)
        item_specific_embedding = self.item_specific_model(features)
        item_general_embedding = self.item_general_model(features)
        return user_embedding, item_specific_embedding, item_general_embedding

    def compute_loss(self, features, training=False):
        user_embedding, item_specific_embedding, item_general_embedding = \
            self.call(features)

        specific_task_loss = self.specific_task_layer(
            user_embedding, item_specific_embedding,
            compute_metrics=not training,
            compute_batch_metrics=True
        )
        general_task_loss = self.general_task_layer(
            user_embedding, item_general_embedding,
            compute_metrics=not training,
            compute_batch_metrics=True
        )
        loss = self.task_weighting_layer(
            tf.stack(
                [
                    specific_task_loss,
                    general_task_loss
                ],
                axis=-1
            )
        )

        regularizaiton = (
            self.regularizaiton_layer([user_embedding, item_specific_embedding]) +
            self.regularizaiton_layer([user_embedding, item_general_embedding])
        )

        return loss + regularizaiton

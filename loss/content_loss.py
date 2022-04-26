import tensorflow as tf


class ContentLoss(tf.keras.losses.Loss):
    def __init__(self, perceptual=False, *args, **kwargs) -> None:
        super(ContentLoss, self).__init__(*args, **kwargs)
        self.perceptual = perceptual
        
    def calcLoss(self, x, y):
        out = tf.pow(x-y, 2)
        out = tf.reduce_sum(out, axis=(1,2,3))
        return out[:,None]
    
    def call(self, y_true, y_pred):
        loss = []
        for i in range(len(y_true)):
            loss.append(self.calcLoss(y_true[i], y_pred[i]))
        loss = tf.concat(loss, axis=-1)
        if not self.perceptual:
            loss_scale = tf.stop_gradient(tf.nn.softmax(loss, axis=-1))
            loss = loss*loss_scale
        return tf.reduce_sum(loss, axis=-1)
import tensorflow as tf
import os
from glob import glob
from model.model import get_model
from utils.dataset_utils import train_val_split, tf_parse_filename

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 25

TENSORBOARD_DIR = './logs/1'
MODEL_SAVE_PATH = './saved-model'
MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_PATH, 'weights')

def get_bn_momentum(step):
    return min(0.99, 0.5 + 0.0002*step)


def anchor_loss(y_true, y_pred, gamma=0.5):
    pred_prob = tf.math.sigmoid(y_pred)

    # Obtain probabilities at indices of true class
    true_mask = tf.dtypes.cast(y_true, dtype=tf.bool)
    q_star = tf.boolean_mask(pred_prob, true_mask)
    q_star = tf.expand_dims(q_star, axis=1)

    # Calculate bce and add anchor loss coeff where labels equal 0
    loss_bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    M = 1.0 - y_true
    loss_calc = (M * (1.0 + pred_prob - q_star + 0.05)**gamma + (1.0 - M)) * loss_bce

    return tf.math.reduce_mean(loss_calc)


if __name__ == '__main__':
    if not os.path.exists(TENSORBOARD_DIR):
        os.mkdir(TENSORBOARD_DIR)

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        os.mkdir(MODEL_WEIGHTS_PATH)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    
    # Create datasets (.map() after .batch() due to lightweight mapping fxn)
    print('Creating train and val datasets...')
    TRAIN_FILES, VAL_FILES = train_val_split()
    TEST_FILES = glob('ModelNet40/*/test/*.npy')   # only used to get length for comparison
    print('Number of training samples:', len(TRAIN_FILES))
    print('Number of validation samples:', len(VAL_FILES))
    print('Number of testing samples:', len(TEST_FILES))
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.list_files(TRAIN_FILES)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.list_files(VAL_FILES)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.map(tf_parse_filename, num_parallel_calls=AUTOTUNE)
    print('Done!')

    print('Creating model...')
    bn_momentum = tf.Variable(get_bn_momentum(0), trainable=False)
    my_model = get_model(bn_momentum=bn_momentum)
    print('Done!')
    my_model.summary()

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    loss_fxn = anchor_loss

    # Instantiate metric objects
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

    file_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir = TENSORBOARD_DIR, write_graph=True, write_images=True, histogram_freq=1, profile_batch='80,100')
    tensorboard_cbk.set_model(my_model)

    for epoch in range(EPOCHS):
        # Reset metrics
        train_acc_metric.reset_states()
        val_acc_metric.reset_states()

        # Train on batches
        for x_train, y_train in train_ds:
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables autodifferentiation.
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = my_model(x_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fxn(y_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, my_model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, my_model.trainable_weights))

            # Update training metric.
            train_acc_metric(y_train, logits)

            train_acc = train_acc_metric.result()

            for x_val, y_val in val_ds:
                with tf.GradientTape() as val_tape:
                    val_logits = my_model(x_val, training=False)
                    val_probs = tf.math.sigmoid(val_logits)
                    val_acc_metric.update_state(y_val, val_probs)
            val_acc = val_acc_metric.result()

            print('Epoch {:03d}, Accuracy: {:6f}, Loss: {:6f}, Val Accuracy: {:6f}'.format(epoch + 1, train_acc, loss_value, val_acc))

        with file_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=epoch)
            tf.summary.scalar('accuracy', train_acc, step=epoch)
            tf.summary.scalar('val_accuracy', val_acc, step=epoch)

    # my_model.save(MODEL_SAVE_PATH)
    my_model.save_weights(MODEL_WEIGHTS_PATH, save_format='tf')

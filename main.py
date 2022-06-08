import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data import load_data, split_data, GetStats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

data_df = load_data(10)
train_df, test_df = split_data(data_df, 1, 1)
tr_df, val_df = train_test_split(train_df, train_size=0.6, random_state=0)

tr_X, tr_y = tr_df.drop(columns='score'), tr_df['score'].to_numpy()
val_X, val_y = val_df.drop(columns='score'), val_df['score'].to_numpy()
te_X, te_y = test_df.drop(columns='score'), test_df['score'].to_numpy()

pipe = Pipeline([
    ('stats', GetStats(n_jobs=-1)),
    ('scaler', StandardScaler()),
])
tr_X = pipe.fit_transform(tr_X[['yes_data', 'no_data']].itertuples(index=False))
val_X = pipe.transform(val_X[['yes_data', 'no_data']].itertuples(index=False))
te_X = pipe.transform(te_X[['yes_data', 'no_data']].itertuples(index=False))

batch_size = 16
input_width = tr_X.shape[1]

tr_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(tr_X, dtype=tf.float32)).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(val_X, dtype=tf.float32)).batch(batch_size)

def get_clf_ds(X, y):
    idx1 = np.arange(X.shape[0])
    np.random.shuffle(idx1)
    idx2 = np.arange(X.shape[0])
    np.random.shuffle(idx2)
    clf_ds_1 = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X[idx1], dtype=tf.float32))
    clf_ds_2 = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X[idx2], dtype=tf.float32))
    clf_ds_y = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(((y[idx1]-y[idx2]) > 0).reshape(-1, 1), dtype=tf.float32))
    clf_ds = tf.data.Dataset.zip((clf_ds_1, clf_ds_2, clf_ds_y)).batch(batch_size)
    return clf_ds

clf_tr_ds = get_clf_ds(tr_X, tr_y)
clf_val_ds = get_clf_ds(val_X, val_y)
clf_te_ds = get_clf_ds(te_X, te_y)


def dense_layers(
    first_width,
    act,
    act_out,
    kernel_l1,
    kernel_l2,
    bias_l2,
    act_l2,
    depth,
    width_decr_rate,
    width_min,
    width_out,
):
    input_layer = layers.Input(shape=first_width)
    _x = input_layer
    w = first_width
    for _ in range(depth-1):
        _x = layers.Dense(
            w, activation=act,
            kernel_regularizer=tf.keras.regularizers.L1L2(l1=kernel_l1, l2=kernel_l2),
            bias_regularizer=tf.keras.regularizers.L2(l2=bias_l2),
            activity_regularizer=tf.keras.regularizers.L2(l2=act_l2),
        )(_x)
        _x = layers.BatchNormalization()(_x)
        _x = layers.Dropout(0.1)(_x)
        w = max(width_min, int(width_decr_rate*w))
    _x = layers.Dense(
        width_out, activation=act_out,
    )(_x)
    return input_layer, _x


encoder_act = 'elu'
encoder_kernel_l1 = 1e-4
encoder_kernel_l2 = 1e-4
encoder_bias_l2 = 1e-4
encoder_act_l2 = 1e-4
encoder_depth = 32
encoder_width_decr_rate = 0
encoder_width_min = 32
encoder_width_out = 32

encoder_input, encoder_output = dense_layers(input_width,
                 encoder_act,
                 None,
                 encoder_kernel_l1,
                 encoder_kernel_l2,
                 encoder_bias_l2,
                 encoder_act_l2,
                 encoder_depth,
                 encoder_width_decr_rate,
                 encoder_width_min,
                 encoder_width_out,
                 )
encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

scorer_act = 'elu'
scorer_kernel_l1 = 1e-4
scorer_kernel_l2 = 1e-4
scorer_bias_l2 = 1e-4
scorer_act_l2 = 1e-4
scorer_depth = 32
scorer_width_decr_rate = 0
scorer_width_min = 8
scorer_width_out = 8

scorer_input, scorer_output = dense_layers(encoder_width_out+input_width,
                 scorer_act,
                 None,
                 scorer_kernel_l1,
                 scorer_kernel_l2,
                 scorer_bias_l2,
                 scorer_act_l2,
                 scorer_depth,
                 scorer_width_decr_rate,
                 scorer_width_min,
                 scorer_width_out,
                 )

scorer = tf.keras.Model(scorer_input, scorer_output, name="scorer")

clfhead_act = 'relu'
clfhead_kernel_l1 = 1e-4
clfhead_kernel_l2 = 1e-4
clfhead_bias_l2 = 1e-4
clfhead_act_l2 = 1e-4
clfhead_depth = 2
clfhead_width_decr_rate = 1
clfhead_width_min = 4
clfhead_width_out = 1
clfhead_input, clfhead_output = dense_layers(scorer_width_out,
                                             clfhead_act,
                                             None,
                                             clfhead_kernel_l1,
                                             clfhead_kernel_l2,
                                             clfhead_bias_l2,
                                             clfhead_act_l2,
                                             clfhead_depth,
                                             clfhead_width_decr_rate,
                                             clfhead_width_min,
                                             clfhead_width_out,
                                             )
clfhead = tf.keras.Model(clfhead_input, clfhead_output, name="clfhead")

class MyModel(tf.keras.Model):
    def __init__(self, encoder, scorer, clfhead, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)
        self.encoder = encoder
        self.scorer = scorer
        self.clfhead = clfhead
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy = tf.keras.metrics.BinaryAccuracy(
            name="acc"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy,
        ]

    def train_step(self, data):
        ds_one, ds_two, is_gt = data

        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one, training=True), self.encoder(ds_two, training=True)
            z1 = tf.concat([z1, ds_one], axis=1)
            z2 = tf.concat([z2, ds_two], axis=1)
            p1, p2 = self.scorer(z1, training=True), self.scorer(z2, training=True)
            c1, c2 = self.clfhead(p1, training=True), self.clfhead(p2, training=True)
            loss = tf.keras.losses.binary_crossentropy(is_gt, c1-c2, from_logits=True)

        # Compute gradients and update the parameters.
        learnable_params = (
            self.encoder.trainable_variables + self.scorer.trainable_variables + self.clfhead.trainable_variables
        )
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.accuracy.update_state(is_gt, tf.math.sigmoid(c1-c2))
        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        ds_one, ds_two, is_gt = data

        z1, z2 = self.encoder(ds_one, training=False), self.encoder(ds_two, training=False)
        z1 = tf.concat([z1, ds_one], axis=1)
        z2 = tf.concat([z2, ds_two], axis=1)
        p1, p2 = self.scorer(z1, training=False), self.scorer(z2, training=False)
        c1, c2 = self.clfhead(p1, training=False), self.clfhead(p2, training=False)
        loss = tf.keras.losses.binary_crossentropy(is_gt, c1 - c2, from_logits=True)

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(is_gt, tf.math.sigmoid(c1-c2))
        return {m.name: m.result() for m in self.metrics}

    def call(self, data):
        ds_one, ds_two, is_gt = data

        z1, z2 = self.encoder(ds_one, training=False), self.encoder(ds_two, training=False)
        z1 = tf.concat([z1, ds_one], axis=1)
        z2 = tf.concat([z2, ds_two], axis=1)
        p1, p2 = self.scorer(z1, training=False), self.scorer(z2, training=False)
        c1, c2 = self.clfhead(p1, training=False), self.clfhead(p2, training=False)
        return tf.concat([c1, c2], axis=1)


earlystop_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    baseline=None,
    restore_best_weights=True
)


mymodel = MyModel(encoder, scorer, clfhead)
mymodel.compile(optimizer=tf.keras.optimizers.Nadam(0.01), weighted_metrics=[])
history = mymodel.fit(clf_tr_ds, validation_data=clf_val_ds, epochs=100, callbacks=[
    earlystop_callback
])

import tensorflow as tf
from tensorflow_probability import distributions as tfd


def repeat_dist(mean, stddev, batch_size):
    # [dist1, dist2, dist3] -> [dist1, dist2, dist3, dist1, dist2, dist3, dist1, dist2, dist3]
    mean = tf.tile(mean, (1, batch_size, 1))
    stddev = tf.tile(stddev, (1, batch_size, 1))
    return tfd.MultivariateNormalDiag(mean, stddev)


def nce_seq(seq_mean, seq_std, seq_embed, warm_up):
    """
    seq_mean, seq_std, seq_embed: (batch_size, seq_length, state_dim)
    """
    batch_size = seq_embed.shape[0]
    seq_length = seq_embed.shape[1] - warm_up

    seq_mean = tf.transpose(seq_mean, [1, 0, 2])[warm_up:]
    seq_std = tf.transpose(seq_std, [1, 0, 2])[warm_up:]
    seq_embed = tf.transpose(seq_embed, [1, 0, 2])[warm_up:]

    seq_embed_pred = repeat_dist(seq_mean, seq_std, batch_size)
    seq_embed = tf.repeat(seq_embed, batch_size, axis=1)

    # scores[i, j]
    scores = seq_embed_pred.log_prob(seq_embed)
    scores = tf.reshape(scores, (seq_length, batch_size, batch_size))

    # score(i,i)
    numerator = tf.linalg.diag_part(scores)  # seq_length x batch_size
    # log (1/N sum e^score(j,i)) = b + log (1/N sum e^(score(j,i)-b))
    normalize = tf.reduce_max(tf.stop_gradient(scores), axis=-1, keepdims=True)
    denominator = tf.math.log(tf.reduce_mean(tf.exp(scores - normalize), axis=-1)) + tf.squeeze(normalize)
    return tf.reduce_mean(numerator - denominator)


def iid_nce_seq(seq_embed):
    """
    seq_mean, seq_std, seq_embed: (batch_size, seq_length, state_dim)
    """
    batch_size = seq_embed.shape[0]
    seq_length = seq_embed.shape[1]

    seq_embed = tf.transpose(seq_embed, [1, 0, 2])

    seq_embed_repeat_1 = tf.tile(seq_embed, (1, batch_size, 1))
    seq_embed_repeat_2 = tf.repeat(seq_embed, batch_size, axis=1)

    # scores[i, j]
    #     scores = tf.keras.losses.cosine_similarity(seq_embed_repeat_1, seq_embed_repeat_2, axis=-1)
    scores = tf.reduce_sum(seq_embed_repeat_1 * seq_embed_repeat_2, axis=-1)
    scores = tf.reshape(scores, (seq_length, batch_size, batch_size))

    # score(i,i)
    numerator = tf.linalg.diag_part(scores)  # seq_length x batch_size
    # log (1/N sum e^score(j,i)) = b + log (1/N sum e^(score(j,i)-b))
    normalize = tf.reduce_max(tf.stop_gradient(scores), axis=-1, keepdims=True)
    denominator = tf.math.log(tf.reduce_mean(tf.exp(scores - normalize), axis=-1)) + tf.squeeze(normalize)
    return tf.reduce_mean(numerator - denominator)


def cons_seq(seq_mean, seq_std, seq_embed, warm_up):
    """
    seq_mean, seq_std, seq_embed: (batch_size, seq_length, state_dim)
    """
    seq_embed_pred = tfd.MultivariateNormalDiag(seq_mean[:, warm_up:], seq_std[:, warm_up:])
    cons_loss = tf.reduce_mean(seq_embed_pred.log_prob(seq_embed[:, warm_up:]))
    return cons_loss

from absl import logging

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import tensorflow_hub as hub
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time
import functools


module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")


def time_eval(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print("--- %s seconds ---" % (time.time() - start_time))
        return result
    return wrapped


input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
with tf.io.gfile.GFile(spm_path, mode="rb") as f:
    sp.LoadFromSerializedProto(f.read())
print("SentencePiece model loaded at {}.".format(spm_path))


def process_to_IDs_in_sparse_format(sp, sentences):
    # An utility method that processes sentences with the sentence piece processor
    # 'sp' and returns the results in tf.SparseTensor-similar format:
    # (values, indices, dense_shape)

    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]

    return (values, indices, dense_shape)


def plot_similarity(labels, labels_to_check, features, features_to_check, rotation):
    corr = np.inner(features, features_to_check)
    sns.set(font_scale=0.8)
    f, ax = plt.subplots(figsize=(16, 12))
    g = sns.heatmap(
        corr,
        xticklabels=labels_to_check,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        ax=ax,
        cmap="YlOrRd")
    g.set_xticklabels(labels_to_check, rotation=rotation)
    g.set_title("Semantic Textual Similarity")


def run_and_plot(session, input_placeholder, messages, to_check):
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)
    values_to_check, indices_to_check, dense_shape_to_check = process_to_IDs_in_sparse_format(sp, to_check)

    message_embeddings = session.run(
        encodings,
        feed_dict={input_placeholder.values: values,
                   input_placeholder.indices: indices,
                   input_placeholder.dense_shape: dense_shape})
    message_to_check = session.run(
        encodings,
        feed_dict={input_placeholder.values: values_to_check,
                   input_placeholder.indices: indices_to_check,
                   input_placeholder.dense_shape: dense_shape_to_check})

    plot_similarity(messages, to_check, message_embeddings, message_to_check, 90)


messages = [
    # Smartphones
    "I like my phone",
    "My phone is not good.",
    "Your cellphone looks great.",

    # Weather
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",

    # Food and health
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",

    # Asking about age
    "How old are you?",
    'Whats your age again?'
]

test = [
    'Whats your age again?',
]
""" "The path of the righteous man is beset on all sides",
    "By the inequities of the selfish and the tyranny of evil men",
    "Blessed is he who, in the name of charity and good will",
    "Shepherds the weak through the valley of darkness",
    "For he is truly his brother's keeper and the finder of lost children",
    "And I will strike down upon thee",
    "With great vengeance and furious anger",
    "Those who attempt to poison and destroy my brothers",
    "And you will know my name is the Lord",
    "When I lay my vengeance upon thee","""

# with tf.Session() as session:
#    session.run(tf.global_variables_initializer())
#    session.run(tf.tables_initializer())
#    run_and_plot(session, input_placeholder, messages, test)
#    plt.show()


# ------------------------------ Cosine similarities -------------------------------------------------------------
import math
sts_input1 = tf.sparse_placeholder(tf.int64, shape=(None, None))
sts_input2 = tf.sparse_placeholder(tf.int64, shape=(None, None))

# For evaluation we use exactly normalized rather than
# approximately normalized.
sts_encode1 = tf.nn.l2_normalize(
    module(
        inputs=dict(values=sts_input1.values,
                    indices=sts_input1.indices,
                    dense_shape=sts_input1.dense_shape)),
    axis=1)
sts_encode2 = tf.nn.l2_normalize(
    module(
        inputs=dict(values=sts_input2.values,
                    indices=sts_input2.indices,
                    dense_shape=sts_input2.dense_shape)),
    axis=1)
cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
sim_scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi

values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)
values_to_check, indices_to_check, dense_shape_to_check = process_to_IDs_in_sparse_format(sp, test)


@time_eval
def run_sts_benchmark(session):
    """Returns the similarity scores"""
    scores = session.run(
        sim_scores,
        feed_dict={
            sts_input1.values: values,
            sts_input1.indices: indices,
            sts_input1.dense_shape: dense_shape,
            sts_input2.values: values_to_check,
            sts_input2.indices: indices_to_check,
            sts_input2.dense_shape: dense_shape_to_check,
        })
    return scores


def get_similar():
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        scores = run_sts_benchmark(session)

        return [v for i, v in enumerate(messages) if scores[i] >= 0.90]

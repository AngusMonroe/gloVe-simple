#!/usr/bin/env python

from argparse import ArgumentParser
import codecs
from collections import Counter
from functools import partial
import logging
from math import log
import os.path
import pickle as p
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse


logger = logging.getLogger("glove")


def get_or_build(path, build_fn, *args, **kwargs):
    """
    从序列化的形式加载或建立一个对象，保存内置的目的。
    剩余的参数被提供给`build_fn`。
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        with open(path, 'rb') as obj_f:
            obj = msgpack.load(obj_f, use_list=False, encoding='utf-8')
    else:
        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)
        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj


def build_vocab(corpus):  # 使用语料建立词典
    """
    为整个语料库建立一个带有词频的词汇表。
    返回一个字典`w - >（i，f）`，将字串映射为对单词ID和单词语料库频率。
    """

    logger.info("Building vocab from corpus")

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    logger.info("Done building vocab from corpus.")

    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


def build_cooccur(vocab, corpus, window_size=10, min_count=None):
    """
    为给定的语料库建立一个单词共现列表。

     这个函数是一个元组生成器，其中每个元素（表示
     一个共生对）就是这种形式

         （i_main，i_context，cooccurrence）

     其中`i_main`是共同发生的主要词的ID
     `i_context`是上下文词的ID，'cooccurrence`是
     如Pennington等人所述的“X_ {ij}”共生值。

     如果`min_count`不是`None`，那么同时出现的是其中的任何一个词
     发生在语料库中的时间少于'min_count'次数被忽略。
    """

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    # Collect cooccurrences internally as a sparse matrix for passable
    # indexing speed; we'll convert into a list later
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)
    # lil_matrix则是使用两个列表存储非0元素。data保存每行中的非零元素,rows保存非零元素所在的列
    # 这种格式也很适合逐个添加元素，并且能快速获取行相关的数据。

    # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            logger.info("Building cooccurrence matrix: on line %i", i)

        tokens = line.strip().split()  # 以空格为标准分出每个单词
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):
            # 将所有单词ID收集在中心词的左侧窗口中
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                distance = contexts_len - left_i  # 与中心词之间的距离

                increment = 1.0 / float(distance)  # 单词之间距离的倒数加权

                # 对称地构建共生矩阵（假设我们也在计算正确的上下文）
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # 产生元组序列（挖掘到LiL-matrix内部来快速遍历所有非零单元格）
    for i, (row, data) in enumerate(zip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def run_iter(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    使用给定的cooccurrence数据和先前计算的权重向量/偏差和伴随的梯度历史来运行GloVe训练的单次迭代。

     `data`是一个预先获取的数据/权重列表，其中每个元素都是这种形式

         （v_main，v_context，
          b_main，b_context，
          gradsq_W_main，gradsq_W_context，
          gradsq_b_main，gradsq_b_context，
         cooccurrence）

     如`train_glove`函数所产生的。 这个元组中的每个元素都是包含它的数据结构的'ndarray`视图。

     有关`W`形状的信息，请参阅`train_glove`函数，
     `biasses`，`gradient_squared`，`gradient_squared_biases`以及它们应该如何初始化。

     参数`x_max`，`alpha`定义了计算两个字对的成本时的加权函数; 有关更多详细信息，请参阅GloVe纸张。

     返回与给定重量分配相关的成本，并在线更新权重。
    """

    global_cost = 0

    # 随机迭代数据，以免无意中偏向单词向量内容
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # 计算成本函数的内部成分，用于总成本计算和梯度计算
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))

        # 计算损失函数
        #
        #   $$ J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        # 为全局损失追踪器添加加权成本
        global_cost += 0.5 * cost

        # 计算词向量梯度
        #
        # 注意：`main_word`只是`W`的视图（不是副本），所以我们这里的修改会影响全局权重矩阵
        # 同样适用于context_word, biases等.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # 计算偏差项的梯度
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # 执行自适应更新
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
                gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def train_glove(vocab, cooccurrences, iter_callback=None, vector_size=100,
                iterations=25, **kwargs):
    """
    Train GloVe vectors on the given generator `cooccurrences`, where
    each element is of the form

        (word_i_id, word_j_id, x_ij)

     其中`x_ij`是一个共生值$ X_ {ij} $

     如果`iter_callback`不是`None`，则提供的函数将会是
     在每次迭代之后用迄今为止学习的“W”矩阵调用。

     关键字参数被传递到迭代步骤函数
    `run_iter`。

     返回计算的字矢量矩阵`W`。
    """

    vocab_size = len(vocab)

    # 字向量矩阵。 这个矩阵的大小是（2V）*d，其中N是语料库词汇的大小，d是词向量的维数。
    # 所有元素都在范围（-0.5，0.5）中随机初始化，我们为每个单词构建两个单词向量：一个单词作为主（中心）单词，另一个单词作为上下文单词。

    # 由用户决定如何处理所产生的两个向量。
    # 为每个单词添加或平均这两个词，或丢弃上下文向量。
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

    # 偏置项，每项与单个矢量相关联。 一个大小为$ 2V $的数组，在范围（-0.5,0.5）内随机初始化。
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # 训练通过自适应梯度下降（AdaGrad）完成。 为了做到这一点，我们需要存储所有先前渐变的平方和。
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # 将所有平方梯度和初始化为1，这样我们的初始自适应学习率就是全局学习率。
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # 偏差项的平方梯度之和。
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    # 从给定的cooccurrence generator生成可重复使用的列表，预取所有必要的数据。
    #
    # 注意：这些都是实际数据矩阵的视图，因此它们的更新将传递给真实的数据结构
    #
    # （我们甚至将单元素偏置提取为切片，以便我们将它们用作视图）
    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        logger.info("\tBeginning iteration %i..", i)

        cost = run_iter(vocab, data, **kwargs)

        logger.info("\t\tDone (cost %f)", cost)

        if iter_callback is not None:
            iter_callback(W)

    return W


def save_model(W, path):
    with open(path, 'wb') as vector_f:
        p.dump(W, vector_f, protocol=2)

    logger.info("Saved vectors to %s", path)


def main(corpus_path, save_path):
    corpus = open(corpus_path, 'r', encoding='utf-8')

    logger.info("Fetching vocab..")
    vocab = get_or_build(None, build_vocab, corpus)
    logger.info("Vocab has %i elements.\n", len(vocab))

    logger.info("Fetching cooccurrence list..")
    corpus.seek(0)

    # 将generator转换为list
    cooccurrences = list(get_or_build(None, build_cooccur, vocab, corpus, window_size=10, min_count=10))
    logger.info("Cooccurrence list fetch complete (%i pairs).\n",
                len(cooccurrences))

    iter_callback = partial(save_model, path=save_path)

    logger.info("Beginning GloVe training..")
    W = train_glove(vocab, cooccurrences,
                    iter_callback=iter_callback,
                    vector_size=100,
                    iterations=25,
                    learning_rate=0.05)

    # TODO shave off bias values, do something with context vectors
    return W

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s\t%(message)s")
    corpus_path = 'data/S0010938X1500195X.txt'
    save_path = 'data/SaveModel.txt'
    W = main(corpus_path, save_path)
    save_model(W, save_path)

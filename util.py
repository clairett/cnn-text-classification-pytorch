import torchtext.data as data
import torchtext.datasets as datasets
import mydatasets
import gensim
import numpy as np

def get_fields():
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    return text_field, label_field


def get_train_dev(callable, batch_size=32, **kargs):
    text_field, label_field = get_fields()

    train_data, dev_data = callable(text_field, label_field, **kargs)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data),
                                batch_sizes=(batch_size, len(dev_data)),
                                **kargs)

    return text_field, label_field, train_iter, dev_iter, None

def get_train_dev_test(callable, batch_size=50, **kargs):
    text_field, label_field = get_fields()

    train_data, dev_data, test_data = callable(text_field, label_field, **kargs)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(batch_size, len(dev_data), len(test_data)),
                                        **kargs)

    return text_field, label_field, train_iter, dev_iter, test_iter


def get_train_dev_test_from_tsv(batch_size=50, **kargs):
    text_field, label_field = get_fields()

    train_data, dev_data, test_data = data.TabularDataset.splits(path='./data/SST2/', train='train.tsv',
                                                  validation='dev.tsv', test='qualtrics.tsv', format='tsv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data),
                                        batch_sizes=(batch_size, len(dev_data), len(test_data)), sort_key=lambda x: len(x.text),
                                        **kargs)

    return text_field, label_field, train_iter, dev_iter, test_iter


# load MR dataset
def mr(shuffle=False, **kargs):
   return get_train_dev(
         lambda text_field, label_field, **ka: mydatasets.MR.splits(text_field, label_field, shuffle=shuffle),
         **kargs)

# load SST2 dataset
def sst2(**kargs):
    return get_train_dev_test_from_tsv(**kargs)

# load SST dataset
def sst(fine_grained=False, train_subtrees=False, **kargs):
    return get_train_dev_test(
          lambda text_field, label_field, **ka: datasets.SST.splits(text_field, label_field, fine_grained=fine_grained, train_subtrees=train_subtrees),
          **kargs)

def get_unk_vector(dim, range=0.25):
   return np.random.uniform(-range, range, dim).astype("float32")

def load_word_vectors(filepath, binary, vocab):
    model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=binary)
    dim = model.__dict__['vector_size']
    word_vector_matrix = []
    for word in vocab.itos:
        if word in model.vocab:
           word_vector_matrix.append(model.word_vec(word))
        else:
           word_vector_matrix.append(get_unk_vector(dim))

    return np.array(word_vector_matrix)

def print_time(text, timeMeter):
    print("\nTime for {}: {:.2f}{}".format(text, timeMeter.value(), timeMeter.unit))

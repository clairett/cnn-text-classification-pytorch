#! /usr/bin/env python3
import os
import argparse
import datetime
import torch
import torchtext.data as data
import model
import train
import sys
import csv
from util import sst, sst2, mr, load_word_vectors
from collections import OrderedDict
import numpy as np

def load_data(args):
    if args.dataset is None:
       return None, None, None, None, None

    # load data
    print("\nLoading data...")

    split_args = {
          'batch_size': args.batch_size,
          'device': args.device,
          'repeat': False,
          'shuffle':  args.shuffle if args.shuffle else None
          }

    if args.dataset == 'mr':
       return mr(**split_args)
    elif args.dataset == 'sst':
       return sst(fine_grained=args.fine_grained, train_subtrees=args.train_subtrees, **split_args)
    elif args.dataset == 'sst2':
        return sst2(**split_args)


def main():
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=25, help='number of epochs for train [default: 25]')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-save-interval', type=int, default=0, help='how many steps to wait before saving [default:0]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
    # model
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel [default: 100]')
    parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-predictfile', type=str, default=None, help='predict sentences in a file')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('-dataset', type=str, default='sst', help='specify dataset: sst | sst2 | mr | none')
    parser.add_argument('-fine-grained', action='store_true', default=False, help='use 5-class sst')
    parser.add_argument('-train-subtrees', action='store_true', default=False, help='train sst subtrees')
    parser.add_argument('-load-word-vectors', type=str, default=None, help='load pre-trained word vectors in binary format')
    parser.add_argument('-load-saved-word-vectors', type=str, default=None, help='load saved word vectors')
    parser.add_argument('-debug', action='store_true', default=False, help='debug mode')
    args = parser.parse_args()

    # update args and print
    args.dataset = args.dataset if args.dataset != 'none' else None
    text_field, label_field, train_iter, dev_iter, test_iter = load_data(args)

    if train_iter:
       print("train dataset size:", len(train_iter.dataset))
    if dev_iter:
       print("dev dataset size:", len(dev_iter.dataset))
    if test_iter:
       print("test dataset size:", len(test_iter.dataset))

    if args.dataset:
        args.embed_num = len(text_field.vocab)
        args.class_num = len(label_field.vocab) - 1 # exclude <unk>

    word_vector_matrix = None
    if text_field and args.load_word_vectors:
       print("\nLoading pre-trained word vectors...")
       word_vector_matrix = load_word_vectors(args.load_word_vectors, binary=True, vocab=text_field.vocab)

       word_vectors_filepre = os.path.splitext(os.path.basename(args.load_word_vectors))[0]
       np.save('-'.join([args.dataset, word_vectors_filepre, 'word-vectors.npy']), word_vector_matrix)
    elif args.load_saved_word_vectors:
       print("\nLoading saved word vectors...")
       word_vector_matrix = np.load(args.load_saved_word_vectors)

    args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    if args.dataset:
       args.save_dir = os.path.join(args.save_dir, args.dataset, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
       args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # model
    if args.snapshot is None:
        cnn = model.CNN_Text(args, text_field, label_field, word_vector_matrix)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            cnn = torch.load(args.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist."); exit()

    if args.cuda:
        cnn = cnn.cuda()

    print()

    # train or predict
    if args.predict is not None:
        label = train.predict(args.predict, cnn, args)
        print('[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    elif args.predictfile is not None:
        filepre = os.path.splitext(os.path.basename(args.predictfile))[0]
        predictions_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predictions')
        result_path = os.path.join(predictions_dir, filepre + '-predictions-' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +'.csv')
        if not os.path.isdir(predictions_dir): os.makedirs(predictions_dir)
        with open(args.predictfile, 'r') as rf, \
             open(result_path, 'w') as wf:
           writer = csv.writer(wf)
           writer.writerow(['text', 'label'])
           for i, line in enumerate(rf):
              line = line.strip()
              label = train.predict(line, cnn, args)
              writer.writerow([line, label])
              sys.stdout.write('\rPredicted [{}] sentences...'.format(i + 1))
           print('\nPredictions are written to ', result_path)

    elif args.test:
        if test_iter:
            train.eval(test_iter, cnn, args, print_info=True)
        else:
            print("\nThe test dataset does not exist.\n")
    else:
        train.train(train_iter, dev_iter, cnn, args)
        print()

main()

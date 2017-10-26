import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from nltk import word_tokenize
import copy

def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    model.train()
    best_dev_accuracy = 0
    best_model = copy.deepcopy(model)
    best_epoch = 1
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            model.renorm_fc(args.max_norm)
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.data[0],
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if args.save_interval != 0 and steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

        dev_accuracy = eval(dev_iter, model, args, print_info=True)
        if dev_accuracy > best_dev_accuracy:
           best_dev_accuracy = dev_accuracy
           best_model = copy.deepcopy(model)
           best_epoch = epoch

    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    torch.save(best_model, os.path.join(args.save_dir, 'model.pt'))
    print("Best epoch:", best_epoch)
    print("Best dev accuracy:", best_dev_accuracy)


def eval(data_iter, model, args, print_info=False):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = avg_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    if print_info:
       print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                          accuracy,
                                                                          corrects,
                                                                          size))
    return accuracy

def predict(text, model, text_field, label_field, args):
    assert isinstance(text, str)
    model.eval()
    text = word_tokenize(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if args.cuda:
       x = x.cuda()
    #print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_field.vocab.itos[predicted.data[0]+1]

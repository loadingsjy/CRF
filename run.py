import argparse
import random
import datetime
import profile
import CRF_opt
import data
import config

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create Conditional Random Field(CRF) for POS Tagging.'
    )
    parser.add_argument('--bigdata', '-b',
                        action='store_true', default=False,
                        help='use big data')
    parser.add_argument('--anneal', '-a',
                        action='store_true', default=False,
                        help='use simulated annealing')
    # parser.add_argument('--optimize', '-o',
    #                     action='store_true', default=False,
    #                     help='use feature extracion optimization')
    parser.add_argument('--regularize', '-r',
                        action='store_true', default=False,
                        help='use L2 regularization')
    parser.add_argument('--seed', '-s',
                        action='store', default=1000, type=int,
                        help='set the seed for generating random numbers')
    # parser.add_argument('--file', '-f',
    #                     action='store', default='crf.pkl',
    #                     help='set where to store the model')
    args = parser.parse_args()

    print("\nSet the seed for generating random numbers to %d" % args.seed)
    print()
    random.seed(args.seed)

    # 根据参数读取配置
    config = config.Config(args.bigdata)
    begin = datetime.datetime.now()
    print("Loading the dataset...")

    data = data.Dataset(config.train_data_file, config.dev_data_file, config.test_data_file)
    train_data, train_word_num = data.read_data(data.data_type[0])
    dev_data, dev_word_num = data.read_data(data.data_type[1])
    test_data, test_word_num = data.read_data(data.data_type[2])
    print('\t train set has %d sentences, %d words' % (len(train_data), train_word_num))
    print('\t dev set has %d sentences, %d words' % (len(dev_data), dev_word_num))
    print('\t test set has %d sentences, %d words' % (len(test_data), test_word_num))
    print()

    print("Creating Conditional Random Field for POS Tagging...")
    crf = CRF_opt.CRF(train_data, dev_data)
    print('tags num：', crf.N)
    crf.create_partial_feature_space()
    print('feature space：', crf.g)
    print()
    # profile.run("glm.create_partial_feature_space()")
    crf.SGD_training(config.save_file, config.epochs, config.interval, config.C, config.eta, config.batch_size,
                     config.shuffle, args.regularize, args.anneal)
    # profile.run("crf.SGD_training()")

    print("testing on best model...")
    test_model = CRF_opt.CRF.load(config.save_file)
    correct, total, precision = test_model.evaluate(test_data)
    print('test precision:{:,} / {:,} = {:3%}'.format(correct, total, precision))

    end = datetime.datetime.now()
    print("\ntotal using time：{} ".format(end - begin))

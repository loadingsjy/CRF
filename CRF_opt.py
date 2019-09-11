import time
import random
import numpy as np
from scipy.special import logsumexp  # 计算元素指数总和的对数
from collections import defaultdict
import pickle


class CRF(object):
    BOS = "_START_"

    def __init__(self, train, dev):
        self.train = train
        self.dev = dev
        tags = set()  # 词性集合
        self.train_word_num = 0
        self.dev_word_num = 0
        for sentence in self.train:
            for word, tag in sentence:
                tags.add(tag)
            self.train_word_num += len(sentence)
        for sentence in self.dev:
            self.dev_word_num += len(sentence)
        self.tags_list = list(tags)
        self.BOS = "_START_"
        # self.tags_list.sort()
        self.tags = {tag: index for index, tag in enumerate(self.tags_list)}
        self.N = len(self.tags)  # 词性的个数
        self.partial_feature_space = {}
        self.g = len(self.partial_feature_space)
        self.dim = self.g * self.N
        self.w = np.zeros((self.g, self.N))
        self.bi_features = []
        self.bi_scores = []
        self.forward_log_scores = []
        self.backward_log_scores = []
        self.g = defaultdict(float)

    def bi_partial_feature_template(self, pre_tag):
        feature_set = set()
        feature_set.add("".join(["01:", pre_tag]))
        return feature_set

    def uni_partial_feature_template(self, sentence, i):
        con = '_consecutive_'
        pre = "^^"  # 句首标志
        post = "$$"  # 句末标志
        word = sentence[i][0]
        word_len = len(word)
        feature_set = set()
        sentence_len = len(sentence)

        if sentence_len == 1:
            pre_word = pre
            post_word = post
        else:
            if i == 0:
                pre_word = pre
                post_word = sentence[i + 1][0]
            elif i == sentence_len - 1:
                pre_word = sentence[i - 1][0]
                post_word = post
            else:
                pre_word = sentence[i - 1][0]
                post_word = sentence[i + 1][0]

        # feature_set.add('01:' + tag + '*' + pre_tag)
        feature_set.add("".join(['02:', word]))
        feature_set.add("".join(['03:', pre_word]))
        feature_set.add("".join(['04:', post_word]))
        feature_set.add("".join(['05:', word, '*', pre_word[-1]]))
        feature_set.add("".join(['06:', word, '*', post_word[0]]))
        feature_set.add("".join(['07:', word[0]]))
        feature_set.add("".join(['08:', word[-1]]))

        for k in range(1, word_len - 1):
            feature_set.add("".join(['09:', word[k]]))
            feature_set.add("".join(['10:', word[0], '*', word[k]]))
            feature_set.add("".join(['11:', word[-1], '*', word[k]]))

        if word_len == 1:
            feature_set.add(
                "".join(['12:', word, '*', pre_word[-1], '*', post_word[0]]))
        for k in range(word_len - 1):
            if word[k] == word[k + 1]:
                feature_set.add("".join(['13:', word[k], '*', con]))

        for k in range(1, min(5, word_len + 1)):
            feature_set.add("".join(["14:", word[0:k]]))
            feature_set.add("".join(["15:", word[-k:]]))
        return feature_set

    def create_partial_feature_template(self, sentence, i, pre_tag):
        feature_set = self.uni_partial_feature_template(sentence, i) | self.bi_partial_feature_template(
            pre_tag)
        return feature_set

    def create_partial_feature_space(self):  # 构建部分特征空间
        partial_feature_set = set()
        for sentence in self.train:
            sentence_len = len(sentence)
            for i in range(sentence_len):
                if i == 0:
                    pre_tag = self.BOS
                else:
                    pre_tag = sentence[i - 1][1]
                feature_set = self.create_partial_feature_template(sentence, i, pre_tag)
                partial_feature_set |= feature_set
        for index, feature in enumerate(partial_feature_set):
            self.partial_feature_space[feature] = index
        # self.partial_feature_space = {feature: index for index, feature in enumerate(feature_set)}
        self.g = len(self.partial_feature_space)
        self.dim = self.g * self.N
        self.w = np.zeros((self.g, self.N), dtype='float64')
        self.bi_features = [list(self.bi_partial_feature_template(pre_tag)) for pre_tag in self.tags]
        self.bi_scores = np.array(
            [self.cal_score(bi_feature) for bi_feature in self.bi_features])
        # print(self.feature_space, '\n', len(self.feature_space))

    def f_sentence(self, sentence, tags_id_list):  # 将句子标为tags词性序列的对应的所有特征向量
        f_sentence = []
        sentence_len = len(sentence)
        '''if len(sentence) != len(tags_id_list):
            print('tag_list的长度为：', len(tags_id_list))
            print('句子长度：', len(sentence))
            print('长度不对应！')
            return'''
        for i in range(sentence_len):
            # tag = self.tags_list[tags_id_list[i]]
            if i != 0:
                pre_tag = self.tags_list[tags_id_list[i - 1]]
            else:
                pre_tag = self.BOS
            f_sentence.append(list(self.create_partial_feature_template(sentence, i, pre_tag)))
        return f_sentence

    def SGD_training(self, save_file, epochs, interval, c, init_eta, batch_size, shuffle, regularization, anneal):
        b = 0
        self.g = defaultdict(float)
        max_dev_precision = 0.0
        max_iterator = -1
        now_train_precision = 0.0
        count = 0
        # _lambda = 10 * c / self.train_word_num
        # t0 = 1 / (_lambda * init_eta)
        # t = 0
        decay = 0.96
        decay_steps = len(self.train) / batch_size
        global_step = 1
        eta = init_eta
        # eta = 1 / (_lambda * (t0 + t * 20))
        print("start training...")
        if regularization:
            print("use L2 regularization：C=%f" % c)
        if anneal:
            print("ues Simulated annealing\n")
        print('initial eta=%f' % eta)

        total_time = 0.0
        epoch = 0
        for epoch in range(epochs):
            start = time.time()
            print(' epoch：%d ' % (epoch + 1))
            if anneal:
                print(' \t updated eta: {:.4f}'.format(eta))
            # _decay *= (1.0 - _eta * _lambda)
            # _gain = _eta / _decay
            # random.seed(iterator + 1)
            if shuffle:
                random.shuffle(self.train)
                print(' \t data shuffled')
            # print(" \t _decay；", _decay)
            # print(' \t _gain:', _gain)
            for sentence in self.train:
                '''sentence_len = len(sentence)
                correct_tag_id_list = [self.tags.get(tag) for (word, tag) in sentence]
                for features, tag_id in zip(self.f_sentence(sentence, correct_tag_id_list), correct_tag_id_list):
                    for f in features:
                        self.g[(self.partial_feature_space[f], tag_id)] += 1

                self.forward(sentence)
                self.backward(sentence)
                log_z = logsumexp(self.forward_log_scores[-1])

                features = self.create_partial_feature_template(sentence, 0, self.BOS)
                feature_index = [self.partial_feature_space[feature] for feature in features]
                prob = np.exp(self.cal_score(features) + self.backward_log_scores[0] - log_z)
                for index in feature_index:
                    self.g[index] -= prob

                for i in range(1, sentence_len):
                    unigram_features = self.uni_partial_feature_template(sentence, i)
                    uni_index = [self.partial_feature_space[feature] for feature in unigram_features]
                    scores = self.bi_scores + self.cal_score(unigram_features) + self.forward_log_scores[i - 1][:,
                                                                                 np.newaxis] + self.backward_log_scores[
                                 i]  # 计算该句子对每个tag的得分
                    probs = np.exp(scores - log_z)
                    for bi_feature, p in zip(self.bi_features, probs):
                        bi_index = [self.partial_feature_space[bi_f] for bi_f in bi_feature]
                        for feature_index in uni_index + bi_index:  # 列表相加表示将列表连接起来
                            # print(prob)
                            self.g[feature_index] -= p'''
                self.update_gradient(sentence)
                # print(self.bi_scores)
                b += 1
                '''for i in range(len(sentence)):
                    for pre_tag in self.tags_list:
                        p = np.exp(self.cal_log_p(sentence, i, pre_tag) - z)
                        temp = p * [self.partial_feature_space[feature] for feature in
                                   self.create_partial_feature_template(sentence, i, pre_tag)]'''
                if b == batch_size:
                    # self.w += g
                    # print(g)
                    if regularization:
                        self.w *= (1 - c * eta)
                    for index, value in self.g.items():
                        # print(index, value)
                        self.w[index] += value * eta
                    # print(self.w)
                    if anneal is True:
                        eta = init_eta * (decay ** (global_step / decay_steps))
                        # print(decay**(global_step/decay_steps), eta)
                    global_step += 1
                    b = 0
                    # g = np.zeros(self.dim)
                    self.g = defaultdict(float)
                    self.bi_scores = np.array([self.cal_score(bi_feature) for bi_feature in self.bi_features])

            # evaluate after one epoch
            train_correct_num, total_num, train_precision = self.evaluate(self.train)
            print(' \t train precision：{:,} / {:,} = {:.4%}'.format(train_correct_num, total_num, train_precision))
            dev_correct_num, dev_num, dev_precision = self.evaluate(self.dev)
            print(' \t dev precision：{:,} / {:,} = {:.4%}'.format(dev_correct_num, dev_num, dev_precision))

            if dev_precision > max_dev_precision:
                now_train_precision = train_precision
                max_dev_precision = dev_precision
                max_iterator = epoch
                count = 0
                f = open(save_file, 'r+')
                f.truncate()  # 清空文件内容
                f.close()
                self.save(save_file)
            else:
                count += 1
            stop = time.time()
            t = stop - start
            print(" \t this epoch using time ：{:.2f}s ".format(t))
            print('-----------------------------------------------------------')
            total_time += t
            if count > interval:  # interval次迭代后dev准确率没有提升就不再进行迭代训练
                break
        average_time = total_time / (epoch + 1)
        print("\nepoch times: {:2d}".format((epoch + 1)))
        print("average each epoch time {:.2f}s".format(average_time))
        print('epoch{:2d}/{:2d}: maximum dev precision: {:.4%} , train precision: {:.4%}\n'.format((max_iterator + 1),
                                                                                                    (epoch + 1),
                                                                                                    max_dev_precision,
                                                                                                    now_train_precision))

    def update_gradient(self, sentence):
        # self.g = defaultdict(float) 不能初始化为零!!!，当batch size不为1时会把前面的梯度归零
        length = len(sentence)
        tag_list = [tag for word, tag in sentence]
        pre_tag = self.BOS
        for i in range(length):
            cur_tag = tag_list[i]
            cur_tag_index = self.tags.get(cur_tag)
            features = self.create_partial_feature_template(sentence, i, pre_tag)
            for feature in features:
                self.g[(self.partial_feature_space[feature], cur_tag_index)] += 1
            pre_tag = cur_tag
        self.forward(sentence)
        self.backward(sentence)
        z = logsumexp(self.forward_log_scores[-1])

        features = self.create_partial_feature_template(sentence, 0, self.BOS)
        feature_index = [self.partial_feature_space[feature] for feature in features]
        prob = np.exp(self.cal_score(features) + self.backward_log_scores[0] - z)
        for index in feature_index:
            self.g[index] -= prob

        for i in range(1, length):
            unigram_features = self.uni_partial_feature_template(sentence, i)
            unigram_index = [self.partial_feature_space[feature] for feature in unigram_features]
            scores = self.bi_scores + self.cal_score(unigram_features) + self.forward_log_scores[i - 1][:, np.newaxis] + \
                     self.backward_log_scores[i]
            probs = np.exp(scores - z)
            for bigram_feature, prob in zip(self.bi_features, probs):
                bigram_index = [self.partial_feature_space[bi_f] for bi_f in bigram_feature]
                for feature_index in unigram_index + bigram_index:
                    self.g[feature_index] -= prob

    def forward(self, sentence):
        # start = time.time()
        sentence_len = len(sentence)
        self.forward_log_scores = np.zeros((sentence_len, self.N))  # 第 i 个词的词性为 第t个词性 的所有前向词性路径的得分之和
        self.forward_log_scores[0] = self.cal_score(self.create_partial_feature_template(sentence, 0, self.BOS))
        for k in range(1, sentence_len):
            uni_scores = np.array(self.cal_score(self.uni_partial_feature_template(sentence, k)))
            # scores = np.dot(np.exp(bi_scores + uni_scores).T, temp[:, np.newaxis])
            scores = (self.bi_scores + uni_scores).T + self.forward_log_scores[k - 1]
            self.forward_log_scores[k] = logsumexp(scores, axis=1)
        # stop = time.time()
        # print('句子长度：', sentence_len, ' \t forward算法用时：', stop - start)

    def backward(self, sentence):
        # start = time.time()
        sentence_len = len(sentence)
        self.backward_log_scores = np.zeros((sentence_len, self.N))  # 第 i 个词的词性为 第t个词性 的所有后续词性路径的得分之和
        for k in reversed(range(sentence_len - 1)):
            uni_scores = np.array(self.cal_score(self.uni_partial_feature_template(sentence, k + 1)))
            scores = self.bi_scores + uni_scores + self.backward_log_scores[k + 1]
            self.backward_log_scores[k] = logsumexp(scores, axis=1)
        # stop = time.time()
        # print('句子长度：', sentence_len, ' \t backward算法用时：', stop - start)

    def cal_score(self, features):  # 求出该特征对于所有tag的得分
        scores = [self.w[self.partial_feature_space[feature]] for feature in features if
                  feature in self.partial_feature_space]
        return np.sum(scores, axis=0)

    '''def cal_log_p(self, sentence, i, pre_tag):  # 求出所有tag的p值
        return np.array(self.cal_score(self.create_partial_feature_template(sentence, i, pre_tag))) + np.array(
            self.backward_log_scores[i]) + self.forward_log_scores[i - 1, self.tags.get(pre_tag)]'''

    def viterbi(self, sentence):
        # start = time.time()
        sentence_len = len(sentence)
        max_scores = np.zeros((sentence_len, self.N))
        max_index = np.zeros((sentence_len, self.N), dtype='int32')  # max_index[i][t]代表第i个词表为第t个词性的最大得分时前一个词的词性索引
        max_scores[0] = self.cal_score(self.create_partial_feature_template(sentence, 0, self.BOS))
        max_index[0] = -1
        for k in range(1, sentence_len):
            unigram_scores = np.array(self.cal_score(self.uni_partial_feature_template(sentence, k)))
            scores = (self.bi_scores + unigram_scores).T + max_scores[k - 1]
            max_scores[k] = np.max(scores, axis=1)
            max_index[k] = np.argmax(scores, axis=1)
        step = int(np.argmax(max_scores[sentence_len - 1]))
        gold_path = list()
        gold_path.append(step)
        '''for i in range(sentence_len - 1, 0, -1):
            step = max_index[i][step]
            gold_path.insert(0, step)'''
        for i in range(sentence_len - 1):
            step = max_index[sentence_len - 1 - i][step]
            gold_path.insert(0, step)
        # stop = time.time()
        # print('句子长度：', sentence_len, ' \t viterbi算法用时：', stop - start)
        return gold_path

    def evaluate(self, sentences):
        # start = datetime.datetime.now()
        total_num = 0
        correct_num = 0
        for i in range(len(sentences)):
            sentence = sentences[i]
            total_num += len(sentence)
            predict_tag_list = np.array(self.viterbi(sentence))
            correct_tag_list = np.array([self.tags.get(tag) for word, tag in sentence])
            # print(predict_tag_list)
            # print(correct_tag_list)
            correct_num += np.sum(predict_tag_list == correct_tag_list)
            '''for predict_tag, correct_tag in zip(predict_tag_list, correct_tag_list):
                if predict_tag == correct_tag:
                    correct_num += 1'''
        # stop = datetime.datetime.now()
        # print(' \t 评估用时：', str(stop - start))
        return correct_num, total_num, correct_num / total_num

    def save(self, file):
        with open(file, 'wb') as f:
            f.truncate()  # 清空文件内容
            pickle.dump(self, f)

    @classmethod
    def load(cls, file):
        with open(file, 'rb') as f:
            crf = pickle.load(f)
        return crf

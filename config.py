class Config(object):

    def __init__(self, big_data=False):
        if big_data:
            self.interval = 10
            self.batch_size = 50
            self.eta = 0.2
            self.train_data_file = r'big_data\train.conll'
            self.dev_data_file = r'big_data\dev.conll'
            self.test_data_file = r'big_data\test.conll'
        else:
            self.interval = 5
            self.batch_size = 1
            self.eta = 0.5
            self.train_data_file = r'small_data\train.conll'
            self.dev_data_file = r'small_data\dev.conll'
            self.test_data_file = r'small_data\test.conll'

        self.shuffle = True          # 是否打乱数据
        self.regularization = False  # 是否正则化
        self.C = 0.00001             # 正则化系数
        self.save_file = r'.\save'   # 保存模型数据文件
        self.anneal = True           # 是否模拟退火
        self.epochs = 100            # 迭代次数
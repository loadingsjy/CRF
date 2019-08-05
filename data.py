
class Dataset(object):
    data_type = ['train', 'dev', 'test']

    def __init__(self, train_filename, dev_filename, test_filename):
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.tags = set()
        self.tags_num = 0
        self.tags_dict = dict()
        self.words_dict = dict()

    def read_data(self, datatype):
        if datatype == self.data_type[0]:
            filename = self.train_filename
        elif datatype == self.data_type[1]:
            filename = self.dev_filename
        else:
            filename = self.test_filename

        fp = open(filename, 'r', encoding='utf-8')
        sentence = []
        sentences = []
        word_num = 0
        words_set = set()
        while True:
            line = fp.readline()
            if not line:
                break
            else:
                if line != '\n' and line[0] != ' ':
                    word = line.split()[1]
                    tag = line.split()[3]
                    word_num += 1
                    if datatype == self.data_type[0]:
                        self.tags.add(tag)
                        words_set.add(word)
                    sentence.append(tuple((word, tag)))
                else:
                    sentences.append(sentence)
                    sentence = []
        fp.close()
        if datatype == self.data_type[0]:
            self.tags_num = len(self.tags)
            self.tags_dict = {tag: index for index, tag in enumerate(self.tags)}
            self.words_dict = {word: index for index, word in enumerate(words_set)}
        return sentences, word_num   # sentences format：[[(word,tag),(,)(,)...],[],[],[]...[]]

import collections
import re
from d2l import torch as d2l
from collections import defaultdict, Counter


def normalize_text(line, keep_case=False, keep_digits=True):
    """
    文本标准化处理：
    1. 非字母（/数字）字符左右添加空格作为分割边界
    2. 合并连续空白字符为单个空格，去除首尾空白
    3. 可选：保留大小写/保留数字

    :param line: 输入的单行文本字符串
    :param keep_case: 是否保留大小写（默认False：转为小写）
    :param keep_digits: 是否保留数字（默认True：保留；False则数字也作为分割边界）
    :return: 标准化后的文本字符串
    """
    # 步骤0：定义正则匹配模式（根据是否保留数字调整）
    if keep_digits:
        # 匹配：非字母、非数字、非空白的字符（作为分割边界）
        pattern = r'([^A-Za-z0-9\s])'
    else:
        # 匹配：非字母、非空白的字符（数字也会被分割）
        pattern = r'([^A-Za-z\s])'

    # 步骤1：非字母/数字字符左右添加空格（分割边界）
    line_with_spaces = re.sub(pattern, r' \1 ', line)

    # 步骤2：合并连续空白字符合并为单个空格，去除首尾空白
    normalized = re.sub(r'\s+', ' ', line_with_spaces).strip()

    # 步骤3：大小写处理（可选配置）
    if not keep_case:
        normalized = normalized.lower()

    return normalized



# 加载处理数据集
def read_time_machine():  # @save
    """将时间机器数据集加载到文本行的列表中"""
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [normalize_text(line) for line in lines]


class Vocab:  # @save
    """文本词表"""

    def __init__(self, tokens=None, vocab_size=1000, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 展平
        if isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]

        # 训练并对vocab分词，加入'</w>'
        self.vocab = set()
        self.merge_rules = []
        self.word_splits = {}
        self.train_bpe(tokens, vocab_size)

        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token in self.vocab:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def train_bpe(self, corpus, vocab_size=1000):
        """
        训练 BPE 模型。
        返回 (bpe_vocab, word_splits)：
          - bpe_vocab  : 子词集合（所有合并结果）
          - word_splits: 每个单词最终的分割状态，用于精确 apply
        """
        word_freq = Counter(corpus)

        for word in word_freq:
            chars = list(word) + ['</w>']
            self.word_splits[word] = chars
            self.vocab.update(chars)
        cnt = 0
        while len(self.vocab) < vocab_size:
            cnt += 1
            # 打印进度条
            progress = len(self.vocab) / max(vocab_size, 1)
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "█" * filled + "-" * (bar_len - filled)
            print(f"\rBPE训练进度: |{bar}| {len(self.vocab)}/{vocab_size}", end="", flush=True)

            # 统计相邻字符对频率
            pair_freq = defaultdict(int)
            for word, freq in word_freq.items():
                chars = self.word_splits[word]
                for i in range(len(chars) - 1):
                    pair_freq[(chars[i], chars[i + 1])] += freq

            if not pair_freq:
                break

            best_pair = max(pair_freq, key=pair_freq.get)
            new_token = ''.join(best_pair)
            self.merge_rules.append((best_pair, new_token))

            # 更新所有单词的分割
            for word in word_freq:
                chars = self.word_splits[word]
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                        new_chars.append(new_token)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                self.word_splits[word] = new_chars

            self.vocab.add(new_token)
        print('\n', cnt)

    @staticmethod
    def apply_bpe(words, merge_rules, word_splits=None):
        """
        对单个单词应用 BPE 分词。
        优先从 word_splits 缓存中取训练时的结果；
        对未见词则按 merge_rules 顺序依次尝试合并（与训练完全一致）。
        """
        if not isinstance(words, (list, tuple)):
            # 训练时见过的单词直接返回，不会有"消失子词"问题
            if word_splits is not None and words in word_splits:
                return word_splits[words]

            # 未见词：按规则顺序合并（重现训练过程）
            chars = list(words) + ['</w>']
            for (a, b), merged in merge_rules:
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and chars[i] == a and chars[i + 1] == b:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            return chars
        return [wd for word in words for wd in Vocab.apply_bpe(word, merge_rules, word_splits)]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0


def count_corpus(tokens):  # @save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1, vocab_size=1000):
    """返回时间机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    corpus_words = [word for line in lines for word in line.split()]
    vocab = Vocab(corpus_words, vocab_size=vocab_size)
    tokens = Vocab.apply_bpe(corpus_words, vocab.merge_rules, vocab.word_splits)
    corpus = vocab[tokens]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def main():
    vocab_size = 1000
    tokens, vocab = load_corpus_time_machine(vocab_size=vocab_size)
    print(f"词元总数: {len(tokens)}")
    print(f"词表大小: {len(vocab)}")
    print(f"前10个词元: {tokens[:10]}")
    print(f"词表前10个词: {vocab.idx_to_token[:10]}")
    test = 'Hello, today is my first time coming to the bulletin area.'
    test = normalize_text(test)
    test = test.split(' ')
    tokens = Vocab.apply_bpe(test, vocab.merge_rules, vocab.word_splits)
    print(tokens)
    print(vocab[tokens])


if __name__ == '__main__':
    main()

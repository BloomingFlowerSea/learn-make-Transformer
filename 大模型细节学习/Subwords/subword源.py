# 导入所需库
from collections import defaultdict, Counter
import re


# BPE 训练函数
def train_bpe(corpus, vocab_size=1000):
    """
    训练 BPE 模型，生成词汇表
    :param corpus: 输入语料库（单词列表）
    :param vocab_size: 目标词汇表大小
    :return: 词汇表（子词集合）
    """
    # 步骤 1：初始化单词为字符序列，添加词尾标记 </w>
    word_freq = Counter(corpus)  # 统计每个单词的频率
    vocab = set()  # 初始化词汇表
    word_splits = {}  # 存储每个单词的当前分割状态

    for word in word_freq:
        # 将单词拆分为字符列表并添加词尾标记
        chars = list(word) + ['</w>']
        word_splits[word] = chars
        vocab.update(chars)  # 添加初始字符到词汇表

    # 步骤 2：迭代合并频率最高的字符对
    while len(vocab) < vocab_size:
        # 统计所有相邻字符对的频率
        pair_freq = defaultdict(int)
        for word, freq in word_freq.items():
            chars = word_splits[word]
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freq[pair] += freq

        if not pair_freq:  # 如果没有字符对可合并，退出
            break

        # 找到频率最高的字符对
        best_pair = max(pair_freq, key=pair_freq.get)
        new_token = ''.join(best_pair)  # 合并为新子词

        # 步骤 3：更新所有单词的分割，合并 best_pair
        for word in word_freq:
            chars = word_splits[word]
            i = 0
            new_chars = []
            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                    new_chars.append(new_token)
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            word_splits[word] = new_chars

        # 将新子词加入词汇表
        vocab.add(new_token)

    return vocab


# BPE 分词函数
def apply_bpe(word, vocab):
    """
    对单个单词应用 BPE 分词
    :param word: 输入单词
    :param vocab: 训练好的词汇表
    :return: 分词后的子词列表
    """
    # 步骤 1：初始化为字符序列并添加词尾标记
    if not word:
        return []
    chars = list(word) + ['</w>']

    # 步骤 2：贪心合并，直到无法合并为止
    while True:
        # 寻找所有可能的相邻字符对
        pairs = [(chars[i], chars[i + 1]) for i in range(len(chars) - 1)]
        # 检查哪些字符对可以合并（存在于词汇表中）
        mergeable = [''.join(pair) for pair in pairs if ''.join(pair) in vocab]

        if not mergeable:  # 如果没有可合并的字符对，退出
            break

        # 选择第一个可合并的子词（贪心策略）
        best_merge = mergeable[0]
        new_chars = []
        i = 0
        while i < len(chars):
            if i < len(chars) - 1 and ''.join(chars[i:i + 2]) == best_merge:
                new_chars.append(best_merge)
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars

    return chars


# 测试代码
def main():
    # 示例语料库
    corpus = ["low", "low", "lower", "lowest", "new", "newer"]
    print("原始语料库:", corpus)

    # 训练 BPE 模型
    vocab_size = 10  # 设置较小的词汇表大小以便演示
    vocab = train_bpe(corpus, vocab_size)
    print("训练得到的词汇表:", sorted(vocab))

    # 应用 BPE 分词
    test_words = ["low", "lowest", "newest"]
    for word in test_words:
        tokens = apply_bpe(word, vocab)
        print(f"单词 '{word}' 分词结果: {tokens}")


if __name__ == "__main__":
    main()

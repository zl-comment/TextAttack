"""

Stopword Modification
--------------------------

"""

import nltk

from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps


class StopwordModification(PreTransformationConstraint):
    """
    StopwordModification 类是一个预转换约束（PreTransformationConstraint），用于限制在文本攻击过程中对停用词（stopwords）进行修改。

    主要功能:
    1. 继承自 PreTransformationConstraint:
       StopwordModification 继承自 PreTransformationConstraint，意味着它是一个用于限制文本转换操作的约束类。

    2. 初始化方法 __init__:
       接受两个参数：stopwords 和 language。如果提供了 stopwords，则使用该集合；否则，使用 nltk 库提供的指定语言的停用词集合（默认为英语）。

    3. 方法 _get_modifiable_indices:
       返回当前文本中可以修改的单词索引。
       遍历 current_text.words，将不是停用词的单词索引添加到 non_stopword_indices 集合中。
       返回 non_stopword_indices，即可以进行修改的非停用词的索引集合。

    4. 方法 check_compatibility:
       检查给定的转换（transformation）是否与该约束兼容。
       该约束只关注单词替换操作，因为包含停用词的短语释义是可以接受的。
       使用 transformation_consists_of_word_swaps 函数检查转换是否仅由单词替换组成。

    5. 功能总结:
       StopwordModification 的主要作用是确保在文本攻击过程中，不对停用词进行修改。这有助于保持文本的语义完整性，因为停用词通常对句子的核心意义影响较小。
       
    通过限制对停用词的修改，这个约束可以帮助提高生成对抗样本的质量，确保对文本的修改集中在更有意义的单词上。
    """
    def __init__(self, stopwords=None, language="english"):
        if stopwords is not None:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = set(nltk.corpus.stopwords.words(language))

    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        
        non_stopword_indices = set()
        for i, word in enumerate(current_text.words):
            if word not in self.stopwords:  # 停用词
                non_stopword_indices.add(i)
        return non_stopword_indices

    def check_compatibility(self, transformation):
        """The stopword constraint only is concerned with word swaps since
        paraphrasing phrases containing stopwords is OK.

        Args:
            transformation: The ``Transformation`` to check compatibility with.
        """
        return transformation_consists_of_word_swaps(transformation)

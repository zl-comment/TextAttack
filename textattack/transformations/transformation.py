"""
Transformation Abstract Class
============================================

"""

from abc import ABC, abstractmethod

from textattack.shared.utils import ReprMixin
import spacy

class Transformation(ReprMixin, ABC):
    """An abstract class for transforming a sequence of text to produce a
    potential adversarial example."""


    def __call__(
        self,
        current_text,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
        return_indices=False,
        return_phrases_indices=False,    #是否返回短语索引
        phrases_indices=None,  #短语索引
        phrase=False, #确认是否启用短语特征
    ):
        """Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls
        ``_get_transformations``.

        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
            shifted_idxs (bool): Whether indices could have been shifted from
                their original position in the text.
            return_indices (bool): Whether the function returns indices_to_modify
                instead of the transformed_texts.
        """
    
        
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words)))
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(
                current_text.convert_from_original_idxs(indices_to_modify)
            )
        

        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(current_text, self)
            # phrases_indices = set(phrases_indices) & constraint(current_text, self)

        
        #增加了返回短语 已经是经过筛选的
        if phrases_indices is None:
            self.nlp = spacy.load("en_core_web_sm")  # Load the spaCy model
            phrases_indices = set()
            doc = self.nlp(current_text.text)

            # 创建一个不包含符号的 token 列表，同时重新分配新索引
            non_punct_tokens_with_new_indices = [
                (token, new_idx) 
                for new_idx, token in enumerate(doc) if token.pos_ != "PUNCT"
            ]
            covered_indices = set()

            # 提取名词短语
            for chunk in doc.noun_chunks:
                # 使用重新分配的索引
                start = [new_idx for token, new_idx in non_punct_tokens_with_new_indices if token.i == chunk.start][0]
                end = [new_idx for token, new_idx in non_punct_tokens_with_new_indices if token.i == chunk.end - 1][0] + 1
                phrases_indices.add((start, end, "noun-phrase"))
                covered_indices.update(range(start, end))

            # 提取动词短语和固定表达式
            for token, new_index in non_punct_tokens_with_new_indices:
                if token.pos_ == "VERB":
                    phrases_indices.add((new_index, new_index + 1, "verb-phrase"))
                    covered_indices.add(new_index)
                elif token.dep_ == "fixed":
                    phrases_indices.add((new_index, new_index + 1, "fixed-expression"))
                    covered_indices.add(new_index)

            # 添加未覆盖的单词
            for token, new_index in non_punct_tokens_with_new_indices:
                if new_index not in covered_indices:
                    phrases_indices.add((new_index, new_index + 1, "single-word"))

            # 按 token 序号排序
            phrases_indices = sorted(phrases_indices, key=lambda x: x[0])

            # 输出提取的短语和单词
            for start, end, phrase_type in phrases_indices:
                phrase = " ".join([token.text for token, _ in non_punct_tokens_with_new_indices[start:end]])
                print(f"Extracted {phrase_type}: '{phrase}'")
        else:
            phrases_indices = set(phrases_indices)
            
        if return_indices:
            return indices_to_modify
        #增加了返回短语的确定
        if return_phrases_indices:
            return phrases_indices

        if phrase is  False:
            transformed_texts = self._get_transformations(current_text, indices_to_modify)
            for text in transformed_texts:
                text.attack_attrs["last_transformation"] = self
            return transformed_texts

        else :
            transformed_texts_phrases = self._get_transformations_phrases(current_text, phrases_indices)
            for text in transformed_texts_phrases:
                text.attack_attrs["last_transformation"] = self
            return transformed_texts_phrases    
        
        

    @abstractmethod
    def _get_transformations(self, current_text, indices_to_modify):
        """Returns a list of all possible transformations for ``current_text``,
        only modifying ``indices_to_modify``. Must be overridden by specific
        transformations.

        Args:
            current_text: The ``AttackedText`` to transform.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_transformations_phrases(self, current_text, phrases_indices):
        """Returns a list of all possible transformations for ``current_text``,
        only modifying ``indices_to_modify``. Must be overridden by specific
        transformations.

        Args:
            current_text: The ``AttackedText`` to transform.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError()
    @property
    def deterministic(self):
        return True

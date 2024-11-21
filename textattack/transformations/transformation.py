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
        phrases_indices_to_order=None,  #短语索引
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
        #增加了返回短语
        if phrases_indices_to_order is  None:
            self.nlp = spacy.load("en_core_web_sm")  # Load the spaCy model
            phrases_indices_to_order = set()
            # 提取名词短语
            for chunk in self.nlp(current_text.text).noun_chunks:
                phrases_indices_to_order.add((chunk.start, chunk.end, "noun-phrase"))

            # 提取动词短语和固定表达式
            for token in self.nlp(current_text.text):
                if token.pos_ == "VERB":
                    phrases_indices_to_order.add((token.i, token.i + 1, "verb-phrase"))
                elif token.dep_ == "fixed":
                    phrases_indices_to_order.add((token.i, token.i + 1, "fixed-expression"))
        else:
            phrases_indices_to_order = set(phrases_indices_to_order)

        for constraint in pre_transformation_constraints:
            indices_to_modify = indices_to_modify & constraint(current_text, self)
            phrases_indices_to_order = phrases_indices_to_order & constraint(current_text, self)

        

        if return_indices:
            return indices_to_modify
        #增加了返回短语的确定
        if return_phrases_indices:
            return phrases_indices_to_order

        transformed_texts = self._get_transformations(current_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
        return transformed_texts

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

    @property
    def deterministic(self):
        return True

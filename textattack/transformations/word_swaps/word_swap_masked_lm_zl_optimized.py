"""
Word Swap by BERT-Masked LM.
-------------------------------
"""

import itertools
import re

import torch

import gc
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils

from .word_swap import WordSwap
import os
from textattack import LocalPathConfig

class WordSwapMaskedLM_zl(WordSwap):

    def _clear_memory(self):
        """Release memory to prevent crashes"""
        torch.cuda.empty_cache()
        gc.collect()
    """Generate potential replacements for a word using a masked language
    model.

    Based off of following papers
        - "Robustness to Modification with Shared Words in Paraphrase Identification" (Shi et al., 2019) https://arxiv.org/abs/1909.02560
        - "BAE: BERT-based Adversarial Examples for Text Classification" (Garg et al., 2020) https://arxiv.org/abs/2004.01970
        - "BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al, 2020) https://arxiv.org/abs/2004.09984
        - "CLARE: Contextualized Perturbation for Textual Adversarial Attack" (Li et al, 2020): https://arxiv.org/abs/2009.07502

    BAE and CLARE simply masks the word we want to replace and selects replacements predicted by the masked language model.

    BERT-Attack instead performs replacement on token level. For words that are consisted of two or more sub-word tokens,
        it takes the top-K replacements for seach sub-word token and produces all possible combinations of the top replacments.
        Then, it selects the top-K combinations based on their perplexity calculated using the masked language model.

    Choose which method to use by specifying "bae" or "bert-attack" for `method` argument.

    Args:
        method (str): the name of replacement method (e.g. "bae", "bert-attack")
        masked_language_model (Union[str|transformers.AutoModelForMaskedLM]): Either the name of pretrained masked language model from `transformers` model hub
            or the actual model. Default is `bert-base-uncased`.
        tokenizer (obj): The tokenizer of the corresponding model. If you passed in name of a pretrained model for `masked_language_model`,
            you can skip this argument as the correct tokenizer can be infered from the name. However, if you're passing the actual model, you must
            provide a tokenizer.
        max_length (int): the max sequence length the masked language model is designed to work with. Default is 512.
        window_size (int): The number of surrounding words to include when making top word prediction.
            For each word to swap, we take `window_size // 2` words to the left and `window_size // 2` words to the right and pass the text within the window
            to the masked language model. Default is `float("inf")`, which is equivalent to using the whole text.
        max_candidates (int): maximum number of candidates to consider as replacements for each word. Replacements are ranked by model's confidence.
        min_confidence (float): minimum confidence threshold each replacement word must pass.
        batch_size (int): Size of batch for "bae" replacement method.
    """

    def __init__(
        self,
        method="bae",
        masked_language_model="bert-base-uncased",
        tokenizer=None,
        max_length=512,
        window_size=float("inf"),
        max_candidates=50,
        min_confidence=5e-4,
        batch_size=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method
        self.max_length = max_length
        self.window_size = window_size
        self.window_size = min(self.window_size, 10)
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.batch_size = min(self.batch_size, 8)
        #加载预训练的模型的tokenizer和model
        if isinstance(masked_language_model, str):
            masked_language_model_cache = LocalPathConfig.BERT_BASE_UNCASED
            if os.path.exists(masked_language_model_cache):  # 如果是本地路径
                print(f"Loading local model from {masked_language_model_cache}")
                self._language_model = AutoModelForMaskedLM.from_pretrained(masked_language_model_cache)
                self._lm_tokenizer = AutoTokenizer.from_pretrained(masked_language_model_cache, use_fast=True)
            else:  # 从Hugging Face加载模型
                print(f"Loading model from Hugging Face: {masked_language_model}")
                self._language_model = AutoModelForMaskedLM.from_pretrained(masked_language_model)
                self._lm_tokenizer = AutoTokenizer.from_pretrained(masked_language_model, use_fast=True)
        else:
            self._language_model = masked_language_model
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self._lm_tokenizer = tokenizer
        self._language_model.to(utils.device)
        self._language_model.half()
        self._language_model.to('cuda')
        self._language_model.eval()
        self.masked_lm_name = self._language_model.__class__.__name__

    def _encode_text(self, text):
        """Encodes ``text`` using an ``AutoTokenizer``, ``self._lm_tokenizer``.

        Returns a ``dict`` where keys are strings (like 'input_ids') and
        values are ``torch.Tensor``s. Moves tensors to the same device
        as the language model.
        """
        encoding = self._lm_tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding.to(utils.device)
        encoding = encoding.to('cuda')

    def _bae_replacement_words(self, current_text, indices_to_modify):
        """使用 BAE 方法获取要替换的单词的替换词。

        参数:
            current_text (AttackedText): 我们要取替换词的文本。
            indices_to_modify (list): 我们要替换的单词的索引列表。
        """
        masked_texts = []
        # 为每个要修改的索引创建掩码版本的文本
        for index in indices_to_modify:
            masked_text = current_text.replace_word_at_index(
                index, self._lm_tokenizer.mask_token
            )
            masked_texts.append(masked_text.text)

        i = 0
        # 2D 列表，其中每个要修改的索引都有一个替换词列表
        replacement_words = []
        while i < len(masked_texts):
            if i % 5 == 0:  # Clear memory every 5 batches
                self._clear_memory()
            # 批量编码掩码文本
            inputs = self._encode_text(masked_texts[i: i + self.batch_size])
            ids = inputs["input_ids"].tolist()
            with torch.no_grad():
                # 从语言模型获取预测
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # 找到掩码标记的索引
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    # 如果未找到掩码标记，则附加一个空列表
                    replacement_words.append([])
                    continue

                # 获取掩码标记的 logits 和概率
                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                ranked_indices = ranked_indices[:20]  # Limit top candidates
                top_words = []
                for _id in ranked_indices:
                    _id = _id.item()
                    word = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    # 检查单词是否为子句，并在必要时去除 BPE 伪影
                    if utils.check_if_subword(
                            word,
                            self._language_model.config.model_type,
                            (masked_index == 1),
                    ):
                        word = utils.strip_BPE_artifacts(
                            word, self._language_model.config.model_type
                        )
                    # 检查单词是否符合替换条件
                    if (
                            mask_token_probs[_id] >= self.min_confidence
                            and utils.is_one_word(word)
                            and not utils.check_if_punctuations(word)
                    ):
                        top_words.append(word)

                    # 如果我们有足够的候选词或概率太低，则停止
                    if (
                            len(top_words) >= self.max_candidates
                            or mask_token_probs[_id] < self.min_confidence
                    ):
                        break

                replacement_words.append(top_words)

            i += self.batch_size

        return replacement_words

    # def _bert_attack_replacement_words(
    #     self,
    #     current_text,
    #     index,
    #     id_preds,
    #     masked_lm_logits,
    # ):
    #     """Get replacement words for the word we want to replace using BERT-
    #     Attack method.
    #
    #     Args:
    #         current_text (AttackedText): Text we want to get replacements for.
    #         index (int): index of word we want to replace
    #         id_preds (torch.Tensor): N x K tensor of top-K ids for each token-position predicted by the masked language model.
    #             N is equivalent to `self.max_length`.
    #         masked_lm_logits (torch.Tensor): N x V tensor of the raw logits outputted by the masked language model.
    #             N is equivlaent to `self.max_length` and V is dictionary size of masked language model.
    #     """
    #     # We need to find which BPE tokens belong to the word we want to replace
    #     masked_text = current_text.replace_word_at_index(
    #         index, self._lm_tokenizer.mask_token
    #     )
    #     current_inputs = self._encode_text(masked_text.text)
    #     current_ids = current_inputs["input_ids"].tolist()[0]
    #     word_tokens = self._lm_tokenizer.encode(
    #         current_text.words[index], add_special_tokens=False
    #     )
    #
    #     try:
    #         # Need try-except b/c mask-token located past max_length might be truncated by tokenizer
    #         masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
    #     except ValueError:
    #         # If mask-token not found, return empty list
    #         return []
    #
    #     # List of indices of tokens that are part of the target word
    #     target_ids_pos = list(
    #         range(masked_index, min(masked_index + len(word_tokens), self.max_length))
    #     )
    #
    #     if not len(target_ids_pos):
    #         return []
    #     elif len(target_ids_pos) == 1:
    #         # Word to replace is tokenized as a single word
    #         top_preds = id_preds[target_ids_pos[0]].tolist()
    #         replacement_words = []
    #         for id in top_preds:
    #             token = self._lm_tokenizer.convert_ids_to_tokens(id)
    #             if utils.is_one_word(token) and not utils.check_if_subword(
    #                 token, self._language_model.config.model_type, index == 0
    #             ):
    #                 replacement_words.append(token)
    #         return replacement_words
    #     else:
    #         # Word to replace is tokenized as multiple sub-words
    #         top_preds = [id_preds[i] for i in target_ids_pos]
    #         products = itertools.product(*top_preds)
    #         combination_results = []
    #         # Original BERT-Attack implement uses cross-entropy loss
    #         cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
    #         target_ids_pos_tensor = torch.tensor(target_ids_pos)
    #         word_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)
    #         for bpe_tokens in products:
    #             for i in range(len(bpe_tokens)):
    #                 word_tensor[i] = bpe_tokens[i]
    #
    #             logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
    #             loss = cross_entropy_loss(logits, word_tensor)
    #             perplexity = torch.exp(torch.mean(loss, dim=0)).item()
    #             word = "".join(
    #                 self._lm_tokenizer.convert_ids_to_tokens(word_tensor)
    #             ).replace("##", "")
    #             if utils.is_one_word(word):
    #                 combination_results.append((word, perplexity))
    #         # Sort to get top-K results
    #         sorted(combination_results, key=lambda x: x[1])
    #         top_replacements = [
    #             x[0] for x in combination_results[: self.max_candidates]
    #         ]
    #         return top_replacements
    def _bert_attack_replacement_words(
        self,
        current_text,
        index,
        id_preds,
        masked_lm_logits,
    ):
        """使用BERT-Attack方法获取要替换的单词的替换词。
        
        参数:
            current_text (AttackedText): 我们希望获取替换词的文本，包含被攻击的单词和上下文。
            index (int): 要替换的单词在当前文本中的索引位置。
            id_preds (torch.Tensor): 一个N x K的张量，包含由掩蔽语言模型预测的每个标记位置的前K个id。
                N表示文本的最大长度，即`self.max_length`，K表示每个位置的候选单词数量。
            masked_lm_logits (torch.Tensor): 一个N x V的张量，包含掩蔽语言模型输出的原始logits。
                N表示文本的最大长度，V是掩蔽语言模型的词典大小，logits用于计算替换单词的可能性。
        """
        # 1. 找到需要替换的单词在当前文本中的BPE标记。
        # 将要替换的单词替换为掩蔽标记（mask token）。
        masked_text = current_text.replace_word_at_index(
            index, self._lm_tokenizer.mask_token
        )

        # 2. 编码掩蔽后的文本以获取输入ID。
        current_inputs = self._encode_text(masked_text.text)
        current_ids = current_inputs["input_ids"].tolist()[0]  # 获取编码后的input_ids

        # 3. 将当前文本中要替换的单词进行编码以获取其BPE标记。
        word_tokens = self._lm_tokenizer.encode(
            current_text.words[index], add_special_tokens=False
        )

        try:
            # 4. 尝试找到掩蔽标记在输入ID中的索引位置。
            # 如果掩蔽标记位于最大长度之外，则可能会导致截断，因此使用try-except处理。
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            # 如果没有找到掩蔽标记，返回空列表表示没有可替换的单词。
            return []

        # 5. 创建目标单词的标记索引列表，计算该单词在编码文本中的位置。
        target_ids_pos = list(
            range(masked_index, min(masked_index + len(word_tokens), self.max_length))
        )

        # 6. 检查目标标记位置是否存在
        if not len(target_ids_pos):
            # 如果没有找到目标标记位置，返回空列表。
            return []
        elif len(target_ids_pos) == 1:
            # 7. 如果目标单词被标记化为单个标记。
            top_preds = id_preds[target_ids_pos[0]].tolist()  # 获取该位置的前K个预测ID
            replacement_words = []  # 用于存储有效的替换单词
            for id in top_preds:
                # 将ID转换为单词
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                # 检查该单词是否是一个完整的单词，并且不是子词
                if utils.is_one_word(token) and not utils.check_if_subword(
                        token, self._language_model.config.model_type, index == 0
                ):
                    replacement_words.append(token)  # 将有效的替换单词加入列表
            return replacement_words
        else:
            # 8. 如果目标单词被标记化为多个子词。
            top_preds = [id_preds[i] for i in target_ids_pos]  # 获取目标标记位置的预测ID
            products = itertools.product(*top_preds)  # 计算所有可能的BPE标记组合
            combination_results = []  # 存储组合结果
            # 原始BERT-Attack实现使用交叉熵损失来评估组合的有效性
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.tensor(target_ids_pos)  # 转换为张量
            word_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)  # 用于存储BPE标记的张量

            for bpe_tokens in products:
                for i in range(len(bpe_tokens)):
                    word_tensor[i] = bpe_tokens[i]  # 将当前BPE标记存入word_tensor

                # 9. 从masked_lm_logits中选择与目标标记位置对应的logits。
                logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
                loss = cross_entropy_loss(logits, word_tensor)
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()  # 计算困惑度
                word = "".join(
                    self._lm_tokenizer.convert_ids_to_tokens(word_tensor)
                ).replace("##", "")  # 将BPE标记转换为单词，并去除子词标记符号

                # 10. 检查组合结果是否是一个完整单词。
                if utils.is_one_word(word):
                    combination_results.append((word, perplexity))  # 存储有效组合及其困惑度

            # 11. 对组合结果按困惑度排序，以获取前K个结果。
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]
            ]  # 提取前K个替换单词
            return top_replacements  # 返回替换单词列表
    #对句子的替换
    def _bae_replacement_phrases(self, current_text, start_idx, end_idx):
        """使用 BAE 方法获取要替换的短语的替换词。

        参数:
            current_text (AttackedText): 我们要取替换词的文本。
            start_idx (int): 短语开始的索引位置。
            end_idx (int): 短语结束的索引位置。
        """
        masked_texts = []
        # 为每个要修改的索引创建掩码版本的文本
        try:
            masked_text = current_text.replace_phrase_at_index(
               range(start_idx, end_idx) , [self._lm_tokenizer.mask_token] * (end_idx - start_idx)
            )
        except TypeError as e:
            print(f"Error: {e}")
            print(f"start_idx: {start_idx}")
            print(f"end_idx: {end_idx}")
            print(f"mask_token: {self._lm_tokenizer.mask_token}")
            print(f"mask_token_list: {[self._lm_tokenizer.mask_token] * (end_idx - start_idx)}")
            print(f"current_text: {current_text}")
            raise  # 重新抛出异常以便进一步处理或调试
        masked_texts.append(masked_text.text)

        i = 0
        # 2D 列表，其中每个要修改的索引都有一个替换词列表
        replacement_phrases = []
        while i < len(masked_texts):
            if i % 5 == 0:  # Clear memory every 5 batches
                self._clear_memory()
            # 批量编码掩码文本
            inputs = self._encode_text(masked_texts[i: i + self.batch_size])
            ids = inputs["input_ids"].tolist()
            with torch.no_grad():
                # 从语言模型获取预测
                preds = self._language_model(**inputs)[0]

            for j in range(len(ids)):
                try:
                    # 找到掩码标记的索引
                    masked_index = ids[j].index(self._lm_tokenizer.mask_token_id)
                except ValueError:
                    # 如果未找到掩码标记，则附加一个空列表
                    replacement_phrases.append([])
                    continue

                # 获取掩码标记的 logits 和概率
                mask_token_logits = preds[j, masked_index]
                mask_token_probs = torch.softmax(mask_token_logits, dim=0)
                ranked_indices = torch.argsort(mask_token_probs, descending=True)
                ranked_indices = ranked_indices[:20]  # Limit top candidates
                top_phrases = []
                for _id in ranked_indices:
                    _id = _id.item()
                    phrase = self._lm_tokenizer.convert_ids_to_tokens(_id)
                    # 检查短语是否为子句，并在必要时去除 BPE 伪影
                    if utils.check_if_subword(
                            phrase,
                            self._language_model.config.model_type,
                            (masked_index == 1),
                    ):
                        phrase = utils.strip_BPE_artifacts(
                            phrase, self._language_model.config.model_type
                        )
                    # 检查短语是否符合替换条件
                    if (
                            mask_token_probs[_id] >= self.min_confidence
                            and utils.is_one_word(phrase)
                            and not utils.check_if_punctuations(phrase)
                    ):
                        top_phrases.append(phrase)

                    # 如果我们有足够的候选词或概率太低，则停止
                    if (
                            len(top_phrases) >= self.max_candidates
                            or mask_token_probs[_id] < self.min_confidence
                    ):
                        break

                replacement_phrases.append(top_phrases)

            i += self.batch_size

        return replacement_phrases
    #对句子的替换
    def _bert_attack_replacement_phrases(
            self,
            current_text,
            start_idx,
            end_idx,
            id_preds,
            masked_lm_logits,
    ):
        """使用BERT-Attack方法获取要替换的短语的替换词。
        
        参数:
            current_text (AttackedText): 我们希望获取替换词的文本，包含被攻击的短语和上下文。
            start_idx (int): 短语开始的索引位置。
            end_idx (int): 短语结束的索引位置。
            id_preds (torch.Tensor): 一个N x K的张量，包含由掩蔽语言模型预测的每个标记位置的前K个id。
                N表示文本的最大长度，即`self.max_length`，K表示每个位置的候选单词数量。
            masked_lm_logits (torch.Tensor): 一个N x V的张量，包含掩蔽语言模型输出的原始logits。
                N表示文本的最大长度，V是掩蔽语言模型的词典大小，logits用于计算替换单词的可能性。
        """
        # 1. 找到需要替换的短语在当前文本中的BPE标记。
        # 将要替换的短语替换为掩蔽标记（mask token）。
        try:
            masked_text = current_text.replace_phrase_at_index(
                 range(start_idx, end_idx), [self._lm_tokenizer.mask_token] * (end_idx - start_idx)
            )
        except TypeError as e:
            print(f"Error: {e}")
            print(f"start_idx: {start_idx}")
            print(f"end_idx: {end_idx}")
            print(f"mask_token: {self._lm_tokenizer.mask_token}")
            print(f"mask_token_list: {[self._lm_tokenizer.mask_token] * (end_idx - start_idx)}")
            print(f"current_text: {current_text}")
            raise  # 重新抛出异常以便进一步处理或调试

        print(f"masked_text: {masked_text}")
        # 2. 编码掩蔽后的文本以获取输入ID。
        current_inputs = self._encode_text(masked_text.text)
        print(f"current_inputs: {current_inputs}")
        current_ids = current_inputs["input_ids"].tolist()[0]  # 获取编码后的input_ids
        print(f"current_ids: {current_ids}")

        # 3. 将当前文本中要替换的短语进行编码以获取其BPE标记。
        phrase_tokens = self._lm_tokenizer.encode(
            " ".join(current_text.words[start_idx:end_idx]), add_special_tokens=False
        )
        print(f"phrase_tokens: {phrase_tokens}")

        try:
            # 4. 尝试找到掩蔽标记在输入ID中的索引位置。
            # 如果掩蔽标记位于最大长度之外，则可能会导致截断，因此使用try-except处理。
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            # 如果没有找到掩蔽标记，返回空列表表示没有可替换的短语。
            return []

        # 5. 创建目标短语的标记索引列表，计算该短语在编码文本中的位置。
        target_ids_pos = list(
            range(masked_index, min(masked_index + len(phrase_tokens), self.max_length))
        )
        print(f"target_ids_pos: {target_ids_pos}")

        # 6. 检查目标标记位置是否存在
        if not len(target_ids_pos):
            # 如果没有找到目标标记位置，返回空列表。
            return []
        elif len(target_ids_pos) == 1:
            # 7. 如果目标短语被标记化为单个标记。
            top_preds = id_preds[target_ids_pos[0]].tolist()  # 获取该位置的前K个预测ID
            print(f"top_preds: {top_preds}")
            replacement_phrases = []  # 用于存储有效的替换短语
            for id in top_preds:
                # 将ID转换为短语
                phrase = self._lm_tokenizer.convert_ids_to_tokens(id)
                # 检查该短语是否是一个完整的短语，并且不是子词
                if utils.is_one_word(phrase) and not utils.check_if_subword(
                        phrase, self._language_model.config.model_type, start_idx == 0
                ):
                    replacement_phrases.append(phrase)  # 将有效的替换短语加入列表
            return replacement_phrases
        else:
            # 8. 如果目标短语被标记化为多个子词。
            top_preds = [id_preds[i] for i in target_ids_pos]  # 获取目标标记位置的预测ID
            print(f"top_preds: {top_preds}")
            products = itertools.product(*top_preds)  # 计算所有可能的BPE标记组合
            print(f"products: {list(products)}")
            combination_results = []  # 存储组合结果
            # 原始BERT-Attack实现使用交叉熵损失来评估组合的有效性
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
            target_ids_pos_tensor = torch.tensor(target_ids_pos)  # 转换为张量
            phrase_tensor = torch.zeros(len(target_ids_pos), dtype=torch.long)  # 用于存储BPE标记的张量

            for bpe_tokens in products:
                for i in range(len(bpe_tokens)):
                    phrase_tensor[i] = bpe_tokens[i]  # 将当前BPE标记存入phrase_tensor

                # 9. 从masked_lm_logits中选择与目标标记位置对应的logits。
                logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
                loss = cross_entropy_loss(logits.float(), phrase_tensor)  # Convert to FP32
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()  # 计算困惑度
                phrase = "".join(
                    self._lm_tokenizer.convert_ids_to_tokens(phrase_tensor)
                ).replace("##", "")  # 将BPE标记转换为短语，并去除子词标记符号

                # 10. 检查组合结果是否是一个完整短语。
                if utils.is_one_word(phrase):
                    combination_results.append((phrase, perplexity))  # 存储有效组合及其困惑度
            print(f"combination_results: {combination_results}")

            # 11. 对组合结果按困惑度排序，以获取前K个结果。
            sorted(combination_results, key=lambda x: x[1])
            top_replacements = [
                x[0] for x in combination_results[: self.max_candidates]  # 提取前K个替换短语
            ]
            print(f"top_replacements: {top_replacements}")
            return top_replacements  # 返回替换短语列表

    def _get_transformations_phrases(self, current_text, phrases_indices):
        """
        解析 phrases_indices
        """
        # 将 phrases_indices 转换为列表
        phrases_indices = list(phrases_indices)
        print("DEBUG: phrases_indices:", phrases_indices)
    
        transformed_texts = []
        for start_idx, end_idx, idx_type in phrases_indices:
            print(f"DEBUG: Processing index {start_idx}-{end_idx}, type: {idx_type}")
    
            start_idx = int(start_idx)
            end_idx = int(end_idx)
    
            # 使用索引差值判断是单词还是短语
            if (end_idx - start_idx) == 1:
                # 处理单词
                print(f"DEBUG: Handling single-word at index {start_idx}")
                word_at_index = current_text.words[start_idx]
                print(f"DEBUG: Word at index {start_idx}: {word_at_index}")
                
                # 将单词放入句子中
                sentence = current_text.text.replace(word_at_index, "[MASK]")
                print(f"DEBUG: Sentence with masked word: {sentence}")
            else:
                # 处理短语
                print(f"DEBUG: Handling phrase from index {start_idx} to {end_idx}")
                phrase = " ".join(current_text.words[start_idx:end_idx])
                print(f"DEBUG: Original phrase: {phrase}")
                
                # 将短语放入句子中
                sentence = current_text.text.replace(phrase, "[MASK]")
                print(f"DEBUG: Sentence with masked phrase: {sentence}")
            
            # 编码句子
            current_inputs = self._encode_text(sentence)
            print(f"DEBUG: Encoded input for sentence: {current_inputs}")

            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            print(f"DEBUG: Prediction probabilities shape: {pred_probs.shape}")

            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            print(f"DEBUG: Top probabilities: {top_probs}, Top IDs: {top_ids}")

            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()

            if self.method == "bert-attack":
                print("DEBUG: Using BERT-Attack for replacements.")
                if (end_idx - start_idx) == 1:
                    replacement_words = self._bert_attack_replacement_words(
                        current_text,
                        start_idx,
                        id_preds=id_preds,
                        masked_lm_logits=masked_lm_logits,
                    )
                else:
                    replacement_phrases = self._bert_attack_replacement_phrases(
                        current_text,
                        start_idx,
                        end_idx,
                        id_preds=id_preds,
                        masked_lm_logits=masked_lm_logits,
                    )
            elif self.method == "bae":
                print("DEBUG: Using BAE for replacements.")
                if (end_idx - start_idx) == 1:
                    replacement_words = self._bae_replacement_words(
                        current_text, [start_idx]
                    )[0]
                else:
                    replacement_phrases = self._bae_replacement_phrases(
                        current_text, start_idx, end_idx
                    )

            if (end_idx - start_idx) == 1:
                print(f"DEBUG: Replacement words: {replacement_words}")
                for r in replacement_words:
                    r = r.strip("Ġ")
                    if r != word_at_index:
                        transformed_text = current_text.replace_word_at_index(start_idx, r)
                        print(f"DEBUG: Transformed text with replacement '{r}': {transformed_text}")
                        transformed_texts.append(transformed_text)
            else:
                print(f"DEBUG: Replacement phrases: {replacement_phrases}")
                for replacement_phrase in replacement_phrases:
                    replacement_phrase = replacement_phrase.strip("Ġ")
                    original_phrase = " ".join(current_text.words[start_idx:end_idx])
                    if replacement_phrase != original_phrase:
                        transformed_text = current_text.replace_phrase_at_index(range(start_idx, end_idx), replacement_phrase)
                        print(f"DEBUG: Transformed text with replacement phrase '{replacement_phrase}': {transformed_text}")
                        transformed_texts.append(transformed_text)
    
        print("DEBUG: All transformed texts:", transformed_texts)
        return transformed_texts
    
    def extra_repr_keys(self):
        return [
            "method",
            "masked_lm_name",
            "max_length",
            "max_candidates",
            "min_confidence",
        ]

def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word

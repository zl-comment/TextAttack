"""
Word Swap by BERT-Masked LM.
-------------------------------
"""

import itertools
import re

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from textattack.shared import utils

from .word_swap import WordSwap
import os
from nltk import ngrams
from collections import Counter

import logging
import spacy as spacy
# 加载 spacy 的小型英语模型
nlp = spacy.load('en_core_web_sm')


class WordSwapMaskedLM_zl(WordSwap):
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
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        #加载预训练的模型的tokenizer和model
        if isinstance(masked_language_model, str):
            masked_language_model_cache = "/home/cyh/ZLCODE/google/bert-base-uncased"
            if os.path.exists(masked_language_model_cache):  # 如果是本地路径
                print(f"Loading local model from {masked_language_model_cache} zl")
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
    #         self,
    #         current_text,
    #         index,
    #         id_preds,
    #         masked_lm_logits,
    # ):
    #     """使用 BERT 攻击方法获取要替换单词的替换词，现支持双词替换（bigram）。"""
    #
    #     # 检查是否可以形成双词
    #     if index > 0:  # 确保前面有一个词
    #         bigram_masked_text = current_text.replace_word_at_index(
    #             index - 1, self._lm_tokenizer.mask_token + " " + self._lm_tokenizer.mask_token
    #         )
    #         current_inputs = self._encode_text(bigram_masked_text.text)
    #         current_ids = current_inputs["input_ids"].tolist()[0]
    #
    #         # 找到双词掩码的索引
    #         masked_indices = [
    #             current_ids.index(self._lm_tokenizer.mask_token_id),
    #             current_ids.index(self._lm_tokenizer.mask_token_id,
    #                               current_ids.index(self._lm_tokenizer.mask_token_id) + 1)
    #         ]
    #
    #         top_preds = [id_preds[i] for i in masked_indices]
    #
    #         products = itertools.product(*top_preds)
    #         combination_results = []
    #         cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
    #
    #         for bpe_tokens in products:
    #             word_tensor = torch.tensor(bpe_tokens, dtype=torch.long)
    #             logits = masked_lm_logits[masked_indices]
    #             loss = cross_entropy_loss(logits, word_tensor)
    #             perplexity = torch.exp(torch.mean(loss)).item()
    #
    #             # 将 token ID 转换为词并去除 BPE 伪影
    #             word = " ".join(self._lm_tokenizer.convert_ids_to_tokens(bpe_tokens)).replace("##", "")
    #             if utils.is_one_word(word) and word.isalpha():  # 检查是否为单个有效词
    #                 combination_results.append((word, perplexity))
    #
    #         # 排序以获取前 K 个结果
    #         sorted_results = sorted(combination_results, key=lambda x: x[1])
    #         top_replacements = [x[0] for x in sorted_results[:self.max_candidates]]
    #         return top_replacements
    #
    #     # 单词替换的后备处理
    #     masked_text = current_text.replace_word_at_index(index, self._lm_tokenizer.mask_token)
    #     current_inputs = self._encode_text(masked_text.text)
    #     current_ids = current_inputs["input_ids"].tolist()[0]
    #     word_tokens = self._lm_tokenizer.encode(current_text.words[index], add_special_tokens=False)
    #
    #     try:
    #         masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
    #     except ValueError:
    #         return []
    #
    #     target_ids_pos = list(range(masked_index, min(masked_index + len(word_tokens), self.max_length)))
    #
    #     if not len(target_ids_pos):
    #         return []
    #     elif len(target_ids_pos) == 1:
    #         top_preds = id_preds[target_ids_pos[0]].tolist()
    #         replacement_words = []
    #         for id in top_preds:
    #             token = self._lm_tokenizer.convert_ids_to_tokens(id)
    #
    #             # 检查 token 是否为单个有效词且不是子词
    #             if utils.is_one_word(token) and token.isalpha() and not utils.check_if_subword(token,
    #                                                                                            self._language_model.config.model_type,
    #                                                                                            index == 0):
    #                 replacement_words.append(token)
    #         return replacement_words
    #     else:
    #         top_preds = [id_preds[i] for i in target_ids_pos]
    #         products = itertools.product(*top_preds)
    #         combination_results = []
    #         cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
    #         target_ids_pos_tensor = torch.tensor(target_ids_pos)
    #
    #         for bpe_tokens in products:
    #             word_tensor = torch.tensor(bpe_tokens, dtype=torch.long)
    #             logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
    #             loss = cross_entropy_loss(logits, word_tensor)
    #             perplexity = torch.exp(torch.mean(loss, dim=0)).item()
    #
    #             # 合并 tokens 为一个词并去除 BPE 伪影
    #             word = "".join(self._lm_tokenizer.convert_ids_to_tokens(word_tensor)).replace("##", "")
    #             if utils.is_one_word(word) and word.isalpha():  # 检查是否为单个有效词
    #                 combination_results.append((word, perplexity))
    #
    #         # 排序并返回前 K 个替换词
    #         sorted_results = sorted(combination_results, key=lambda x: x[1])
    #         top_replacements = [x[0] for x in sorted_results[:self.max_candidates]]
    #         return top_replacements
    def _replace_text(self,text, start_idx, length, replacement):
        # 将文本分成两部分：替换部分和其余部分
        before = text.text[:start_idx]
        after = text.text[start_idx + length:]
        # 组合新的文本
        new_text= before + replacement + after
        # 创建并返回一个新的 AttackedText 对象
        return new_text


    # def _bert_attack_replacement_words(
    #         self,
    #         current_text,
    #         start_idx,
    #         id_preds,
    #         masked_lm_logits,
    #         end_idx=None  # 默认 None，用于区分单词和短语的替换
    # ):
    #     """
    #     使用 BERT 攻击方法获取要替换词，包括单词和短语替换。
    #     :param start_idx: 替换位置的起始索引（单词或短语的第一个词索引）。
    #     :param id_preds: 模型预测的 top-k 结果。
    #     :param masked_lm_logits: 掩码语言模型的 logits 结果。
    #     :param end_idx: 如果是短语替换，表示短语结束索引；否则为 None 表示单词替换。
    #     :return: 替换候选词列表。
    #     """
    #
    #     # 如果传入了 end_idx，说明需要替换的是短语
    #     if end_idx is not None and end_idx > start_idx:
    #         # 计算短语长度
    #         phrase_len = end_idx - start_idx + 1
    #
    #         # 替换短语，使用连续的 mask tokens
    #         replacement_phrase = " ".join(["[MASK]"] * phrase_len)
    #         masked_text = self._replace_text(current_text, start_idx, phrase_len,replacement_phrase)
    #     else:
    #         # 单词替换，使用单个 mask token
    #         replacement_word = "[MASK]"
    #         masked_text = self._replace_text(current_text, start_idx, 1,replacement_word)
    #
    #     # 编码 masked_text
    #     current_inputs = self._encode_text(masked_text)
    #     current_ids = current_inputs["input_ids"].tolist()[0]
    #
    #     # 获取所有 mask 的索引
    #     masked_indices = [i for i, token_id in enumerate(current_ids) if token_id == self._lm_tokenizer.mask_token_id]
    #
    #     if not masked_indices:
    #         return []
    #
    #     # 获取每个掩码位置的 top-k 预测结果
    #     top_preds = [id_preds[i] for i in masked_indices]
    #
    #     # 对每个掩码位的 top-k 候选词进行组合（如有多个掩码位）
    #     products = itertools.product(*top_preds)
    #     combination_results = []
    #     cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
    #
    #     for bpe_tokens in products:
    #         word_tensor = torch.tensor(bpe_tokens, dtype=torch.long)
    #         logits = torch.index_select(masked_lm_logits, 0, torch.tensor(masked_indices))
    #         loss = cross_entropy_loss(logits, word_tensor)
    #         perplexity = torch.exp(torch.mean(loss, dim=0)).item()
    #
    #         # 合并 token IDs 为词或短语，去除 BPE 伪影（例如“##”）
    #         replacement = " ".join(self._lm_tokenizer.convert_ids_to_tokens(bpe_tokens)).replace("##", "").strip()
    #
    #         # 检查是否为有效词或短语
    #         if utils.is_valid_word(replacement) and replacement.isalpha():
    #             combination_results.append((replacement, perplexity))
    #
    #     # 根据 perplexity 排序，并返回 top-k 替换词
    #     sorted_results = sorted(combination_results, key=lambda x: x[1])
    #     top_replacements = [x[0] for x in sorted_results[:self.max_candidates]]
    #
    #     return top_replacements
    def _bert_attack_replacement_words(
            self,
            current_text,
            start_idx,
            id_preds,
            masked_lm_logits,
            end_idx=None  # 默认 None，用于区分单词和短语的替换
    ):
        """
        使用 BERT 攻击方法获取要替换词，包括单词和短语替换。
        :param start_idx: 替换位置的起始索引（单词或短语的第一个词索引）。
        :param id_preds: 模型预测的 top-k 结果。
        :param masked_lm_logits: 掩码语言模型的 logits 结果。
        :param end_idx: 如果是短语替换，表示短语结束索引；否则为 None 表示单词替换。
        :return: 替换候选词列表。
        """

        # 如果传入了 end_idx，说明需要替换的是短语
        if end_idx is not None and end_idx > start_idx:
            # 计算短语长度
            phrase_len = end_idx - start_idx + 1

            # 替换短语，使用连续的 mask tokens
            replacement_phrase = " ".join(["[MASK]"] * phrase_len)
            masked_text = self._replace_text(current_text, start_idx, phrase_len, replacement_phrase)

            # 编码 masked_text
            current_inputs = self._encode_text(masked_text)  # 确保使用 .text 获取文本
            current_ids = current_inputs["input_ids"].tolist()[0]

            # 获取所有 mask 的索引
            masked_indices = [i for i, token_id in enumerate(current_ids) if
                              token_id == self._lm_tokenizer.mask_token_id]
            if not masked_indices:
                return []

            top_preds = [id_preds[i] for i in masked_indices]

            products = itertools.product(*top_preds)
            combination_results = []
            cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")

            for bpe_tokens in products:
                word_tensor = torch.tensor(bpe_tokens, dtype=torch.long)
                logits = torch.index_select(masked_lm_logits, 0, torch.tensor(masked_indices))
                loss = cross_entropy_loss(logits, word_tensor)
                perplexity = torch.exp(torch.mean(loss, dim=0)).item()

                # 合并 token IDs 为短语，去除 BPE 伪影（例如“##”）
                replacement = " ".join(self._lm_tokenizer.convert_ids_to_tokens(bpe_tokens)).replace("##", "").strip()

                # 检查是否为有效短语
                if utils.is_valid_word(replacement) and replacement.isalpha():
                    combination_results.append((replacement, perplexity))

        else:
            # 单词替换，使用单个 mask token
            replacement_word = "[MASK]"
            masked_text = self._replace_text(current_text, start_idx, 1, replacement_word)

            # 编码 masked_text
            current_inputs = self._encode_text(masked_text)
            current_ids = current_inputs["input_ids"].tolist()[0]

            try:
                masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
            except ValueError:
                return []

            # 获取单词的 top-k 预测结果
            top_preds = id_preds[masked_index].tolist()
            combination_results = []

            for id in top_preds:
                token = self._lm_tokenizer.convert_ids_to_tokens(id)
                # 检查 token 是否为单个有效词且不是子词
                if utils.is_one_word(token) and token.isalpha():
                    combination_results.append((token, 0))  # 假设单词的 perplexity 为 0

        # 根据 perplexity 排序，并返回 top-k 替换词
        sorted_results = sorted(combination_results, key=lambda x: x[1])
        top_replacements = [x[0] for x in sorted_results[:self.max_candidates]]

        return top_replacements

    def _get_phrases(self, doc):
        """
        使用 spacy 识别名词短语和动词短语，并返回它们在句子中的起始和结束位置。
        """
        phrases = []

        # 提取名词短语（noun phrases）
        for chunk in doc.noun_chunks:
            phrases.append((chunk.start, chunk.end, chunk.text, "NP"))

        # 提取动词短语（verb phrases），这里用动词及其依赖的宾语来代表动词短语
        for token in doc:
            if token.pos_ == "VERB":  # 如果是动词
                verb_phrase = [token.text]
                for child in token.children:
                    if child.dep_ in ("dobj", "prep", "advmod"):  # 处理直接宾语、介词短语等
                        verb_phrase.append(child.text)
                phrases.append((token.i, token.i + len(verb_phrase), " ".join(verb_phrase), "VP"))

        return phrases

    def _get_transformations(self, current_text, indices_to_modify):

        # indices_to_modify = list(indices_to_modify)
        # if self.method == "bert-attack":
        #     # 配置 logging 记录模型不符合规定的输出
        #     logging.basicConfig(filename='/home/cyh/ZLCODE/promptbench/examples/invalid_inputs.log',
        #                         format='%(asctime)s %(filename)s %(levelname)s %(message)s',
        #                         datefmt='%a %d %b %Y %H:%M:%S',
        #                         level=logging.INFO)
        #
        #     # 假设 current_text.text 是待处理的文本
        #     text = current_text.text
        #     # 提取二元组
        #     bigrams = list(ngrams(text.split(), 2))
        #     bigram_counts = Counter(bigrams)
        #     most_common_bigrams = bigram_counts.most_common()
        #
        #     # 创建一个字典以快速查找二元组
        #     bigram_dict = {f"{bigram[0]} {bigram[1]}": i for i, bigram in enumerate(most_common_bigrams)}
        #     logging.info(f"bigram_dict: {bigram_dict}")
        #     indices_to_modify = list(indices_to_modify)
        #     if self.method == "bert-attack":
        #         # 编码当前文本
        #         current_inputs = self._encode_text(text)
        #
        #         # 禁用梯度计算，进行前向传播
        #         with torch.no_grad():
        #             pred_probs = self._language_model(**current_inputs)[0][0]
        #
        #         # 获取每个位置的前max_candidates个预测词汇及其概率
        #         top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
        #         id_preds = top_ids.cpu()
        #         masked_lm_logits = pred_probs.cpu()
        #
        #         transformed_texts = []
        #
        #         # 对于每个需要修改的索引
        #         for i in indices_to_modify:
        #             word_at_index = current_text.words[i]
        #
        #             # 检查当前索引前后是否有形成二元组
        #             if i > 0 and f"{current_text.words[i - 1]} {word_at_index}" in bigram_dict:
        #                 bigram = f"{current_text.words[i - 1]} {word_at_index}"
        #
        #                 replacement_words = self._bert_attack_replacement_words(
        #                     current_text,s
        #                     i - 1,  # 使用第一个词的索引
        #                     id_preds=id_preds,
        #                     masked_lm_logits=masked_lm_logits,
        #                 )
        #                 # 对于每个替换词应用到原文本中
        #                 for r in replacement_words:
        #                     transformed_texts.append(current_text.replace_word_at_index(i - 1, r))
        #             else:
        #                 # 获取替换词（对于单个词）
        #                 replacement_words = self._bert_attack_replacement_words(
        #                     current_text,
        #                     i,
        #                     id_preds=id_preds,
        #                     masked_lm_logits=masked_lm_logits,
        #                 )
        #                 # 将替换词应用到原文本中
        #                 for r in replacement_words:
        #                     r = r.strip("Ġ")
        #                     if r != word_at_index:
        #                         transformed_texts.append(
        #                             current_text.replace_word_at_index(i, r)
        #                         )
        #
        #         # 返回变换后的文本
        #         return transformed_texts
            # 配置 logging 记录模型不符合规定的输出
        logging.basicConfig(filename='/home/cyh/ZLCODE/promptbench/examples/invalid_inputs.log',
                                format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                                datefmt='%a %d %b %Y %H:%M:%S',
                                level=logging.INFO)

        text = current_text.text
        doc = nlp(text)  # 使用 spacy 解析文本

        # 识别名词短语和动词短语
        phrases = self._get_phrases(doc)

        # 记录需要替换的短语的起始和结束位置
        indices_to_modify = []

        for start_idx, end_idx, phrase_text, phrase_type in phrases:
            indices_to_modify.extend(list(range(start_idx, end_idx)))

        logging.info(f"Phrases to modify: {phrases}")

        if self.method == "bert-attack":
            # 编码当前文本
            current_inputs = self._encode_text(text)

            # 禁用梯度计算，进行前向传播
            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]

            # 获取每个位置的前 max_candidates 个预测词汇及其概率
            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()

            transformed_texts = []

            # 对每个短语进行替换
            for start_idx, end_idx, phrase_text, phrase_type in phrases:
                replacement_words = self._bert_attack_replacement_words(
                    current_text,
                    start_idx,  # 使用短语的第一个词的索引
                    id_preds=id_preds,
                    masked_lm_logits=masked_lm_logits,
                    end_idx=end_idx  # 传递短语的结束索引
                )

                # 对每个替换词应用到原文本中
                for r in replacement_words:
                    transformed_texts.append(current_text.replace_phrase(start_idx, end_idx, r))

            # 现在处理单词的替换
            for i in range(len(doc)):
                # 如果单词不在需要修改的短语索引中，则处理单词
                if i not in indices_to_modify:
                    replacement_words = self._bert_attack_replacement_words(
                        current_text,
                        i,  # 当前单词的索引
                        id_preds=id_preds,
                        masked_lm_logits=masked_lm_logits
                    )

                    # 对每个替换词应用到原文本中
                    for r in replacement_words:
                        r = r.strip("Ġ")  # BERT tokenizer 有时会添加 Ġ 字符，移除它
                        if r != current_text.words[i]:  # 保证替换词不与原始单词相同
                            transformed_texts.append(current_text.replace_word_at_index(i, r))

            # 返回变换后的文本
            return transformed_texts


        # 如果方法是 "bae"

        elif self.method == "bae":

            # 获取替换词

            replacement_words = self._bae_replacement_words(
                current_text, indices_to_modify
            )

            transformed_texts = []

            # 遍历每个替换词

            for i in range(len(replacement_words)):
                index_to_modify = indices_to_modify[i]
                word_at_index = current_text.words[index_to_modify]

                # 遍历每个替换词的候选词

                for word in replacement_words[i]:
                    word = word.strip("Ġ")
                    # 检查替换词是否符合条件
                    if (
                            word != word_at_index
                            and re.search("[a-zA-Z]", word)
                            and len(utils.words_from_text(word)) == 1
                    ):
                        # 将替换词应用到原文本中
                        transformed_texts.append(
                            current_text.replace_word_at_index(index_to_modify, word)
                        )
            # 返回变换后的文本
            return transformed_texts
        else:
            # 如果方法未被识别，抛出异常
            raise ValueError(f"Unrecognized value {self.method} for `self.method`.")

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

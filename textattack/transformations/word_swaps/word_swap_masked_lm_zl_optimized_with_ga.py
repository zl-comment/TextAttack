"""
Word Swap by BERT-Masked LM.
-------------------------------
"""

import itertools
import re

import torch
import random
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
        self.mutation_prob=0.01
        self.crossover_prob = 0.5
        self.population_size=20 #基础种群个数
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
        
    def _generate_initial_population(self, id_preds, target_ids_pos, max_length):
        population = []
        print("id_preds:", id_preds)
        print("target_ids_pos:", target_ids_pos)
        print([random.choice(id_preds[i]) for i in target_ids_pos])
        for _ in range(self.population_size):
            individual = [random.choice(id_preds[i]) for i in target_ids_pos]
            population.append(individual)
        print(f"Initial population generated: {population}")  # Debug output
        return population

    def _fitness_function(self, individual, id_preds, target_ids_pos, masked_lm_logits):
        phrase_tensor = torch.tensor(individual, dtype=torch.long)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        target_ids_pos_tensor = torch.tensor(target_ids_pos, dtype=torch.int64)
        logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
        logits = logits.float()
        loss = cross_entropy_loss(logits, phrase_tensor)
        perplexity = torch.exp(torch.mean(loss)).item()
        print(f"Fitness for individual {individual}: {perplexity}")  # Debug output
        return perplexity

    def _select_parents(self, population, id_preds, target_ids_pos, masked_lm_logits):
        fitness_scores = [self._fitness_function(individual, id_preds, target_ids_pos, masked_lm_logits) for individual in population]
        selected_parents = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        print(f"Selected parents: {[x[0] for x in selected_parents[:len(selected_parents) // 2]]}")  # Debug output
        return [x[0] for x in selected_parents[:len(selected_parents) // 2]]

    def _crossover(self, parent1, parent2):
        if random.random() < self.crossover_prob:
            crossover_point = random.randint(1, len(parent1) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            print(f"Crossover between {parent1} and {parent2} at {crossover_point}: {child}")  # Debug output
            return child
        return parent1

    def _mutate(self, individual, id_preds, target_ids_pos):
        if random.random() < self.mutation_prob:
            mutation_point = random.randint(0, len(individual) - 1)
            mutation_value = random.choice(id_preds[mutation_point])
            print(f"Mutating individual {individual} at {mutation_point} with {mutation_value}")  # Debug output
            individual[mutation_point] = mutation_value
        return individual

    def _evolve_population(self, population, id_preds, target_ids_pos, masked_lm_logits):
        parents = self._select_parents(population, id_preds, target_ids_pos, masked_lm_logits)
        new_population = []
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child, id_preds, target_ids_pos)
            new_population.append(child)
            print(f"New child added to population: {child}")  # Debug output
        
        return new_population

    def _get_best_replacement(self, population, id_preds, target_ids_pos, masked_lm_logits):
        fitness_scores = [self._fitness_function(individual, id_preds, target_ids_pos, masked_lm_logits) for individual in population]
        best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])
        print(f"Best individual: {best_individual[0]} with fitness: {best_individual[1]}")  # Debug output
        return self._lm_tokenizer.convert_ids_to_tokens(best_individual[0])

    def _get_best_replacement_phrase(self, population, id_preds, target_ids_pos, masked_lm_logits):
        fitness_scores = [self._fitness_function(individual, id_preds, target_ids_pos, masked_lm_logits) for individual in population]
        best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])
        print(f"Best individual: {best_individual[0]} with fitness: {best_individual[1]}")  # Debug output
        bast_phrase_tensor = torch.zeros(len(best_individual[0]), dtype=torch.long)
        for i in range(len(best_individual[0])):
            bast_phrase_tensor[i]=best_individual[0][i]
        bast_phrase_tokens=self._lm_tokenizer.convert_ids_to_tokens(bast_phrase_tensor)
        bast_phrase=" ".join(token.replace("##", "") for token in bast_phrase_tokens)
        return bast_phrase

    def _ga_replacement(self, current_text, start_idx, end_idx, id_preds, masked_lm_logits):
        """使用遗传算法获取替换单词或短语。

        参数:
            current_text (AttackedText): 我们希望获取替换词的文本，包含被攻击的短语和上下文。
            start_idx (int): 短语开始的索引位置。
            end_idx (int): 短语结束的索引位置。
            id_preds (torch.Tensor): 一个N x K的张量，包含由掩蔽语言模型预测的每个标记位置的前K个id。
            masked_lm_logits (torch.Tensor): 一个N x V的张量，包含掩蔽语言模型输出的原始logits。
        """
        # 创建掩码文本
        if start_idx == end_idx:
            masked_text = current_text.replace_word_at_index(start_idx, self._lm_tokenizer.mask_token)
        else:
            masked_text = current_text.replace_phrase_at_index(
                range(start_idx, end_idx), [self._lm_tokenizer.mask_token] * (end_idx - start_idx)
            )

        current_inputs = self._encode_text(masked_text.text)
        current_ids = current_inputs["input_ids"].tolist()[0]

        if start_idx == end_idx:
        # 编码目标单词或短语
            tokens = self._lm_tokenizer.encode(
            current_text.words[start_idx], add_special_tokens=False
        )
        else:
            tokens = self._lm_tokenizer.encode(
            " ".join(current_text.words[start_idx:end_idx]), add_special_tokens=False
        )

        try:
            masked_index = current_ids.index(self._lm_tokenizer.mask_token_id)
        except ValueError:
            return []

        target_ids_pos = list(range(masked_index, min(masked_index + len(tokens), self.max_length)))

        population = self._generate_initial_population(id_preds, target_ids_pos, self.max_length)

        for _ in range(10):
            population = self._evolve_population(population, id_preds, target_ids_pos, masked_lm_logits)
        if start_idx == end_idx:
            best_replacement = self._get_best_replacement(population, id_preds, target_ids_pos, masked_lm_logits)
        else:
            best_replacement = self._get_best_replacement_phrase(population, id_preds, target_ids_pos, masked_lm_logits)
        
        
        print(f"Best replacement found: {best_replacement}")  # Debug output
        return best_replacement

    def _bert_attack_replacement_words(self, current_text, index, id_preds, masked_lm_logits):
        return self._ga_replacement(current_text, index, index, id_preds, masked_lm_logits)

    def _bert_attack_replacement_phrases(self, current_text, start_idx, end_idx, id_preds, masked_lm_logits):
        return self._ga_replacement(current_text, start_idx, end_idx, id_preds, masked_lm_logits)

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

            # 判断是单词还是短语
            is_single_word = (end_idx - start_idx) == 1
            target_text = current_text.words[start_idx] if is_single_word else " ".join(current_text.words[start_idx:end_idx])
            # print(f"DEBUG: Original {'word' if is_single_word else 'phrase'}: {target_text}")

            # 将目标文本放入句子中，替换为 [MASK]
            mask_tokens = "[MASK]" if is_single_word else " ".join(["[MASK]"] * len(target_text.split()))
            sentence = current_text.text.replace(target_text, mask_tokens)
            # print(f"DEBUG: Sentence with masked {'word' if is_single_word else 'phrase'}: {sentence}")

            # 编码句子
            current_inputs = self._encode_text(sentence)
            # print(f"DEBUG: Encoded input for sentence: {current_inputs}")

            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            # print(f"DEBUG: Prediction probabilities shape: {pred_probs.shape}")

            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            # print(f"DEBUG: Top probabilities: {top_probs}, Top IDs: {top_ids}")

            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()
            # print(f"DEBUG: ID predictions: {id_preds}, Masked LM logits: {masked_lm_logits}")

            # 选择替换方法
            if self.method == "bert-attack":
                replacement_items = self._bert_attack_replacement_words(
                    current_text, start_idx, id_preds=id_preds, masked_lm_logits=masked_lm_logits
                ) if is_single_word else self._bert_attack_replacement_phrases(
                    current_text, start_idx, end_idx, id_preds=id_preds, masked_lm_logits=masked_lm_logits
                )
            elif self.method == "bae":
                replacement_items = self._bae_replacement_words(
                    current_text, [start_idx]
                )[0] if is_single_word else self._bae_replacement_phrases(
                    current_text, start_idx, end_idx
                )

            # print(f"DEBUG: Replacement {'words' if is_single_word else 'phrases'}: {replacement_items}")
            for replacement in replacement_items:
                replacement = replacement.strip("Ġ")
                if replacement != target_text:
                    print(f"DEBUG: Replacing {'word' if is_single_word else 'phrase'} '{target_text}' with '{replacement}'")
                    transformed_text = current_text.replace_word_at_index(start_idx, replacement) if is_single_word else current_text.replace_phrase_at_index(range(start_idx, end_idx), replacement.split())
                    # print(f"DEBUG: Transformed text with replacement '{replacement}': {transformed_text}")
                    transformed_texts.append(transformed_text)

        # print("DEBUG: All transformed texts:", transformed_texts)
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

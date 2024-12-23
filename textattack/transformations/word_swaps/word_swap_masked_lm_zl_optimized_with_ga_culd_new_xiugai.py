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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = "1"
from textattack import LocalPathConfig
from nltk.corpus import wordnet
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
        max_candidates=40,
        min_confidence=5e-4,
        batch_size=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device("cuda:0")
        self.num_iterations = 50  # 优化迭代次数
        self.learning_rate = 0.01  # 学习率
        self.temperature = 0.1  # 温度系数
        self.max_temperature = 10 #最大温度
        self.noise_std = 0.01  # 噪声标准差
        self.diff_weight = 0.5  # 差异损失权重
        self.mutation_prob=0.01
        self.crossover_prob = 0.5
        self.population_size = 50 #基础种群个数
        self.max_iterations = 50 #最大迭代次数
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
            TRAIN_MODEL=LocalPathConfig.BERT_BASE_UNCASED_TRAIN
            if os.path.exists(masked_language_model_cache):  # 如果是本地路径
                #print(f"Loading local model from {masked_language_model_cache}")
                #print(f"Loading local train model from {TRAIN_MODEL}")
                self._language_model = AutoModelForMaskedLM.from_pretrained(masked_language_model_cache,output_attentions=True)
                self._lm_tokenizer = AutoTokenizer.from_pretrained(masked_language_model_cache, use_fast=True)
            else:  # 从Hugging Face加载模型
                #print(f"Loading model from Hugging Face: {masked_language_model}")
                self._language_model = AutoModelForMaskedLM.from_pretrained(masked_language_model)
                self._lm_tokenizer = AutoTokenizer.from_pretrained(masked_language_model, use_fast=True)
        else:
            self._language_model = masked_language_model
            if tokenizer is None:
                raise ValueError(
                    "`tokenizer` argument must be provided when passing an actual model as `masked_language_model`."
                )
            self._lm_tokenizer = tokenizer
        self._language_model.to(self.device)
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
        return encoding.to(self.device)
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
        
    def _generate_initial_population_word(self, id_preds, target_ids_pos, max_length):
        population = []
        #print("id_preds:", id_preds)
        #print("target_ids_pos:", target_ids_pos)

        if len(target_ids_pos) == 1:
            # 如果目标是单个标记
            #print([random.choice(id_preds[i]) for i in target_ids_pos])
            for _ in range(self.population_size):
                individual = [self._select_alphabetic_token(id_preds, i) for i in target_ids_pos]
                population.append(individual)
        else:
            # 如果目标是多个子词
            #print("Generating population for subword tokens:")
            for _ in range(self.population_size):
                individual = []
                for i in target_ids_pos:
                    # 获取当前位置的候选标记并选择其中一个
                    chosen_token = random.choice(id_preds[i])
                    individual.append(chosen_token)
            
                # 合并多个子词为一个完整的单词
                combined_token = self._combine_subwords_to_word(individual)
                population.append(combined_token)

        #print(f"Initial population generated: {population}")  # Debug output
        return population

    def _generate_initial_population_phrase(self, id_preds, target_ids_pos, max_length):
        population = []
        #print("id_preds:", id_preds)
        #print("target_ids_pos:", target_ids_pos)
        #print([random.choice(id_preds[i]) for i in target_ids_pos])
        for _ in range(self.population_size):
            individual = [self._select_alphabetic_token(id_preds, i) for i in target_ids_pos]
            population.append(individual)
        #print(f"Initial population generated: {population}")  # Debug output
        return population


    def _combine_subwords_to_word(self, subwords):
        """将多个子词组合成一个完整的单词。

        如果子词前面有 '##'，则去除它并合并。
        """
        if isinstance(subwords[0], torch.Tensor):
            subwords = [self._lm_tokenizer.decode([subword]) for subword in subwords]
        
        # 返回一个列表，存储所有的 token IDs
        combined_tokens = [self._lm_tokenizer.encode(subword, add_special_tokens=False)[0] for subword in subwords]
        #print("combined_tokens:", combined_tokens)
        # 去除  前缀
        combined_tokens = [token for token in combined_tokens if token != self._lm_tokenizer.bos_token_id]
        #print("combined_tokens:", combined_tokens)

        
        return combined_tokens

    def _fitness_function(self, individual, id_preds, target_ids_pos, masked_lm_logits):
        phrase_tensor = torch.tensor(individual, dtype=torch.long)
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction="none")
        target_ids_pos_tensor = torch.tensor(target_ids_pos, dtype=torch.int64)
        logits = torch.index_select(masked_lm_logits, 0, target_ids_pos_tensor)
        logits = logits.float()
        loss = cross_entropy_loss(logits, phrase_tensor)
        perplexity = torch.exp(torch.mean(loss)).item()
        #print(f"Fitness for individual {individual}: {perplexity}")  # Debug output
        return perplexity

    def _select_parents(self, population, id_preds, target_ids_pos, masked_lm_logits):
        fitness_scores = [self._fitness_function(individual, id_preds, target_ids_pos, masked_lm_logits) for individual in population]
        selected_parents = sorted(zip(population, fitness_scores), key=lambda x: x[1])
        #print(f"Selected parents: {[x[0] for x in selected_parents[:len(selected_parents) // 2]]}")  # Debug output
        return [x[0] for x in selected_parents[:len(selected_parents) // 2]]

    def _crossover(self, parent1, parent2):
        #print("crosspver:", parent1, parent2)
        if random.random() < self.crossover_prob and len(parent1) > 1:
            crossover_point = random.randint(1, len(parent1) - 1)
            
            # 获取两个父代在交叉点的词的ID
            word1_id = parent1[crossover_point]
            word2_id = parent2[crossover_point]
            
            # 将tensor ID转换为实际的词
            word1_str = self._lm_tokenizer.decode([word1_id])
            word2_str = self._lm_tokenizer.decode([word2_id])
            
            # 使用这两个ID创建输入序列
            input_ids = torch.tensor([[word1_id, word2_id]], device=self.device)
            
            # 使用BERT进行预测
            with torch.no_grad():
                outputs = self._language_model(input_ids)
                predictions = outputs.logits
            
            # 获取top_k个预测结果
            top_k = min(10, self.max_candidates)
            predicted_tokens = torch.topk(predictions[0, 0], top_k)  # 预测第一个位置的词
            
            # 解码预测的token为实际的词
            candidates = []
            for token_id in predicted_tokens.indices:
                word = self._lm_tokenizer.decode([token_id])
                if word.strip() and token_id not in [word1_id, word2_id]:  # 使用ID比较
                    candidates.append(token_id.item())  # 保存ID而不是词
            
            # 如果有候选词，随机选择一个
            if candidates:
                new_id = random.choice(candidates)
                child = parent1.copy()
                child[crossover_point] = torch.tensor(new_id)  # 使用tensor形式保存
                #print(f"Crossover generated new word '{self._lm_tokenizer.decode([new_id])}' from parents '{word1_str}' and '{word2_str}'")
                return child
                
        return parent1

    def _mutate(self, individual, id_preds, target_ids_pos):
        if random.random() < self.mutation_prob:
            mutation_point = random.randint(0, len(individual) - 1)
            
            # 获取当前位置的预测概率分布
            predictions = id_preds[target_ids_pos[mutation_point]]
            
            # 确保predictions是tensor并转换为浮点型
            if not isinstance(predictions, torch.Tensor):
                predictions = torch.tensor(predictions)
            predictions = predictions.float()
            
            # 确保predictions是二维的
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(0)
                    
            # 将logits转换为概率
            probabilities = torch.nn.functional.softmax(predictions, dim=-1).squeeze()
            
            # 根据概率分布采样
            try:
                selected_idx = torch.multinomial(probabilities, 1)
                mutation_value = selected_idx.item()  # 直接使用采样的索引作为新的token id
                
                # 确保选择的token是有效的
                decoded_token = self._lm_tokenizer.decode([mutation_value])
                if decoded_token.strip() and not any(token in decoded_token for token in ["[SEP]", "[MASK]", "[PAD]", "[CLS]"]):
                    individual[mutation_point] = mutation_value
            except Exception as e:
                # 如果采样失败，保持原值不变
                pass
                
        return individual

    def _evolve_population(self, population, id_preds, target_ids_pos, masked_lm_logits):
        parents = self._select_parents(population, id_preds, target_ids_pos, masked_lm_logits)
        new_population = []
        
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self._crossover(parent1, parent2)
            child = self._mutate(child, id_preds, target_ids_pos)
            new_population.append(child)
            #print(f"New child added to population: {child}")  # Debug output
        
        return new_population

    def _get_best_replacement(self, population, id_preds, target_ids_pos, masked_lm_logits):
        fitness_scores = [self._fitness_function(individual, id_preds, target_ids_pos, masked_lm_logits) for individual in population]
        best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])
        #print(f"Best individual: {best_individual[0]} with fitness: {best_individual[1]}")  # Debug output
        return self._lm_tokenizer.convert_ids_to_tokens(best_individual[0])

    def _get_best_replacement_phrase(self, population, id_preds, target_ids_pos, masked_lm_logits):
        fitness_scores = [self._fitness_function(individual, id_preds, target_ids_pos, masked_lm_logits) for individual in population]
        best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])
        #print(f"Best individual phrase: {best_individual[0]} with fitness: {best_individual[1]}")  # Debug output
        
        phrase = self._lm_tokenizer.convert_ids_to_tokens(best_individual[0])
        return [' '.join(phrase)]

    def _ga_replacement(self, current_text, start_idx, end_idx, id_preds, masked_lm_logits):
        #print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        """使用遗传算法获取替换单词或短语。

        参数:
            current_text (AttackedText): 我们希望获取替换词的文本，包含被攻击的短语和上下文。
            start_idx (int): 短语开始的索引位置。
            end_idx (int): 短语结束的索引位置。
            id_preds (torch.Tensor): 一个N x K的张量，包含由掩蔽语言模型预测的每个标记位置的前K个id。
            masked_lm_logits (torch.Tensor): 一个N x V的张量，包含掩蔽语言模型输出的原始logits。
        """
        # 创建掩码文本
        
        if  start_idx == end_idx:
            masked_text = current_text.replace_word_at_index(start_idx, self._lm_tokenizer.mask_token)
        else:
            masked_text = current_text.replace_phrase_at_index(
                range(start_idx, end_idx) , [self._lm_tokenizer.mask_token] * (end_idx - start_idx)
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
        # 生成初始种群
        if start_idx == end_idx:
            population = self._generate_initial_population_word(id_preds, target_ids_pos, self.max_length)
        else:
            population = self._generate_initial_population_phrase(id_preds, target_ids_pos, self.max_length)   
        for _ in range(10):
            population = self._evolve_population(population, id_preds, target_ids_pos, masked_lm_logits)
        if start_idx == end_idx:
            best_replacement = self._get_best_replacement(population, id_preds, target_ids_pos, masked_lm_logits)
        else:
            best_replacement = self._get_best_replacement_phrase(population, id_preds, target_ids_pos, masked_lm_logits)
        
        
        #print(f"Best replacement found: {best_replacement}")  # Debug output
        return best_replacement

    def _bert_attack_replacement_words(self, current_text, index, id_preds, masked_lm_logits):
        return self._attack_replacement_words_with_logits(current_text, index, id_preds, masked_lm_logits)

    def _bert_attack_replacement_phrases(self, current_text, start_idx, end_idx, id_preds, masked_lm_logits):
        # 使用logits优化的替换短语选择
        return self._attack_replacement_phrases_with_logits(current_text, start_idx, end_idx, id_preds, masked_lm_logits)

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
        #print("---------phrases_indices:", phrases_indices)
        # 将 phrases_indices 转换为列表并根据 start_idx 进行排序
        phrases_indices = sorted(list(phrases_indices), key=lambda x: x[0])
        #print("DEBUG: phrases_indices:", phrases_indices)

        transformed_texts = []
        for start_idx, end_idx, idx_type in phrases_indices:
            #print(f"DEBUG: Processing index {start_idx}-{end_idx}, type: {idx_type}")

            start_idx = int(start_idx)
            end_idx = int(end_idx)

            # 判断是单词还是短语
            is_single_word = (end_idx - start_idx) == 1
            #print(f"DEBUG: is_single_word: {is_single_word}")
            target_text = current_text.words[start_idx] if is_single_word else " ".join(current_text.words[start_idx:end_idx])
            # #print(f"DEBUG: Original {'word' if is_single_word else 'phrase'}: {target_text}")

            # 将目标文本放入句子中，替换为 [MASK]
            mask_tokens = "[MASK]" if is_single_word else " ".join(["[MASK]"] * len(target_text.split()))
            sentence = current_text.text.replace(target_text, mask_tokens)
            # #print(f"DEBUG: Sentence with masked {'word' if is_single_word else 'phrase'}: {sentence}")

            # 编码句子
            current_inputs = self._encode_text(sentence)
            # #print(f"DEBUG: Encoded input for sentence: {current_inputs}")

            with torch.no_grad():
                pred_probs = self._language_model(**current_inputs)[0][0]
            # #print(f"DEBUG: Prediction probabilities shape: {pred_probs.shape}")

            top_probs, top_ids = torch.topk(pred_probs, self.max_candidates)
            # #print(f"DEBUG: Top probabilities: {top_probs}, Top IDs: {top_ids}")

            id_preds = top_ids.cpu()
            masked_lm_logits = pred_probs.cpu()
            # #print(f"DEBUG: ID predictions: {id_preds}, Masked LM logits: {masked_lm_logits}")

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

            #print(f"DEBUG: Replacement {'words' if is_single_word else 'phrases'}: {replacement_items}")
            for replacement in replacement_items:
                if isinstance(replacement, tuple):
                    replacement = "".join(replacement)  # 或者使用 " ".join(replacement) 如果需要空格分隔
                # 检查是否为不需要的标记
                if any(token in replacement for token in ["[SEP]", "[MASK]"]):
                    continue
                # 检查是否为不需要的标记
                if (utils.is_one_word(replacement) and not utils.check_if_punctuations(replacement)):
                    replacement = replacement.strip("Ġ")
                    if replacement != target_text:
                        #print(f"DEBUG: Replacing {'word' if is_single_word else 'phrase'} '{target_text}' with '{replacement}'")
                        transformed_text = current_text.replace_word_at_index(start_idx, replacement) if is_single_word else current_text.replace_phrase_at_index(range(start_idx, end_idx), replacement.split())
                        # #print(f"DEBUG: Transformed text with replacement '{replacement}': {transformed_text}")
                        transformed_texts.append(transformed_text)

        # #print("DEBUG: All transformed texts:", transformed_texts)
        return transformed_texts
    
    def extra_repr_keys(self):
        return [
            "method",
            "masked_lm_name",
            "max_length",
            "max_candidates",
            "min_confidence",
        ]

    def _select_alphabetic_token(self, id_preds, position):
        """
        Select a random alphabetic token from id_preds at the given position.
        """
        while True:
            token_id = random.choice(id_preds[position])
            token = self._lm_tokenizer.convert_ids_to_tokens([token_id])[0]
            if token.isalpha():
                return token_id

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
        """
        Logits optimization methods for word replacement
        """

    def _attack_replacement_phrases_with_logits(self, current_text, start_idx, end_idx, id_preds, masked_lm_logits):
        """使用logits优化的替换短语选择,结合遗传算法"""
       
        ga_best_replacement = self._ga_replacement(current_text, start_idx, end_idx, id_preds, masked_lm_logits)
        print(f"ga_best_replacement_phrase: {ga_best_replacement}, type: {type(ga_best_replacement)}")
        # 初始化优化器参数
        masked_lm_logits = torch.clamp(masked_lm_logits, min=-1e2, max=1e2).requires_grad_(True)
        epsilon = torch.zeros_like(masked_lm_logits)
        epsilon = torch.clamp(epsilon, min=-1e2, max=1e2)
        optim = torch.optim.Adam([epsilon], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=10, gamma=0.9)

        # 获取原始短语的编码
        original_phrase = " ".join(current_text.words[start_idx:end_idx])
        original_ids = self._lm_tokenizer.encode(original_phrase, add_special_tokens=False)
        original_tensor = torch.tensor(original_ids, device=self.device)

        best_replacements = [ga_best_replacement]
        best_loss = float('inf')

        # 优化循环
        found_valid_replacement = False  # 添加标志来追踪是否找到合适的替换词
        iter = 0
        while not found_valid_replacement and iter < self.num_iterations:
            optim.zero_grad()

            # 计算当前logits并应用温度缩放
            current_logits = (masked_lm_logits + epsilon) / max(self.temperature, 1e-2)  # 温度过小时避免除零
            current_logits = torch.clamp(current_logits, min=-1e2, max=1e2)  # 限制logits范围
            current_probs = torch.nn.functional.softmax(current_logits, dim=-1).detach()
            current_probs.requires_grad = True  # 确保 current_probs 也支持梯度
            # 计算流畅度损失
            flu_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(current_logits, dim=-1),
                torch.nn.functional.softmax(masked_lm_logits / self.temperature, dim=-1),
                reduction='batchmean'
            )

            # 将 original_tensor 转换为 one-hot 编码
            original_onehot = torch.nn.functional.one_hot(original_tensor, num_classes=self._lm_tokenizer.vocab_size).float()
            original_onehot = original_onehot.to(current_probs.device)  # 确保在同一设备上
            original_onehot = torch.clamp(original_onehot, min=1e-6, max=1 - 1e-6)  # 防止极小的概率值
            
            # 检查并调整 repeat_times 的计算方式
            repeat_times = (current_probs.size(0) + original_onehot.size(0) - 1) // original_onehot.size(0)

            # 使用 repeat 方法扩展 original_onehot
            original_onehot_expanded = original_onehot.repeat(repeat_times, 1)
            
            # 确保扩展后的张量与 current_probs 大小匹配
            original_onehot_expanded = original_onehot_expanded[:current_probs.size(0)]
            # 计算余弦相似度
            diff_loss = torch.nn.functional.cosine_similarity(current_probs+ 1e-8, original_onehot_expanded+ 1e-8, dim=-1).mean()
            # 总损失
            loss = flu_loss + self.diff_weight * diff_loss

            if loss < best_loss:
                best_loss = loss.item()
                # 获取当前最佳的替换短语
                top_values, top_indices = torch.topk(current_probs, k=self.max_candidates)

                best_replacements.clear()  # 清空之前的替换词
                for row_indices, row_values in zip(top_indices, top_values):
                    phrase = self._lm_tokenizer.decode(row_indices)
                    # 检查是否为不需要的标记，并且只包含字母
                    if (utils.is_one_word(phrase) and not utils.check_if_punctuations(phrase) and
                        not any(token in phrase for token in ["[SEP]", "[MASK]"]) and
                        phrase.isalpha()):
                        best_replacements.append(phrase)
                        found_valid_replacement = True

            # 反向传播
            loss.backward()
            optim.step()
            scheduler.step()

            # 添加噪声增加多样性
            noise = torch.normal(0, self.noise_std, size=epsilon.size(), device=epsilon.device)
            epsilon.data += noise

            
            # 如果没找到合适的替换词，增加温度系数来扩大搜索范围
            if not found_valid_replacement:
                self.temperature = min(self.temperature * 1.1, self.max_temperature)  # 设置最大温度限制
                # 逐渐增加温度系数
                iter += 1
                # 如果达到最大迭代次数，则终止
                if iter >= self.max_iterations:
                    print("Reached maximum iterations with no valid replacement found.")
                    break

            
        # 确保所有元素都是字符串或元组
        best_replacements = [tuple(replacement) if isinstance(replacement, list) else replacement for replacement in best_replacements]
        
        # 然后去重并限制候选数量

        return list(set(best_replacements))[:self.max_candidates]

    def _attack_replacement_words_with_logits(self, current_text, index, id_preds, masked_lm_logits):
        """修改后的BERT攻击替换方法,集成logits优化"""
        

        # 使用logits优化获取候选词
        logits_replacements = self._get_replacement_words_with_logits(
            current_text, index, id_preds, masked_lm_logits
        )
    
        # 合并两种方法的结果并去重
        all_replacements = list(set(logits_replacements))
    
        replacement_scores = []
        for word in all_replacements:
            
            if isinstance(word, tuple):
                word = "".join(word)  # 或者使用 " ".join(word) 如果需要空格分隔
            # 检查是否为不需要的标记
            if any(token in word for token in ["[SEP]", "[MASK]"]):
                continue
            # 创建替换后的文本
            transformed_text = current_text.replace_word_at_index(index, word)
            inputs = self._encode_text(transformed_text.text)
            
            # 使用原始文本的 input_ids 作为标签
            labels = self._encode_text(current_text.text)["input_ids"]
            inputs["labels"] = labels
            
            # 计算perplexity
            with torch.no_grad():
                outputs = self._language_model(**inputs)
                loss = outputs.loss
                if loss is not None:
                    perplexity = torch.exp(loss).item()
                    replacement_scores.append((word, perplexity))
                else:
                    print("Warning: Model did not return a loss.")
    
        # 按perplexity排序并返回最佳的替换词
        sorted_replacements = sorted(replacement_scores, key=lambda x: x[1])
        return [word for word, _ in sorted_replacements[:self.max_candidates]]

    def _get_replacement_words_with_logits(self, current_text, index, id_preds, masked_lm_logits):
        """使用logits优化的替换词选择,结合遗传算法"""
        # 获取GA的最佳替换词
        ga_best_replacement = self._ga_replacement(current_text, index, index, id_preds, masked_lm_logits)
        print(f"ga_best_replacement_word: {ga_best_replacement}, type: {type(ga_best_replacement)}")
        
        # 如果GA找到了好的替换词，将其作为初始最佳选择
        if ga_best_replacement:
            best_replacements = []
            # 使用GA的结果来初始化epsilon
            if isinstance(ga_best_replacement, list):
                ga_best_replacement = ga_best_replacement[0]  # 取列表中的第一个词
                best_replacements.append(ga_best_replacement)
            ga_word_ids = self._lm_tokenizer.encode(ga_best_replacement, add_special_tokens=False)
            ga_tensor = torch.tensor(ga_word_ids, device=self.device)
            ga_embedding = self._language_model.get_input_embeddings()(ga_tensor).mean(dim=0)
            
            
            
            # 初始化epsilon使其偏向GA的结果
            masked_lm_logits = masked_lm_logits.to(self.device)  # 先移到目标设备
            masked_lm_logits = torch.clamp(masked_lm_logits, min=-1e2, max=1e2).requires_grad_(True)
            # 创建一个临时张量来存储初始值
            temp_epsilon = torch.zeros_like(masked_lm_logits,device=self.device)
            # 然后将临时张量转换为需要梯度的张量
            epsilon = temp_epsilon.detach().clone().requires_grad_(True)

        else:
            # 如果GA没有找到好的替换词，使用原来的初始化方式
            best_replacements = []
            masked_lm_logits = torch.clamp(masked_lm_logits, min=-1e2, max=1e2).requires_grad_(True)
            epsilon = torch.zeros_like(masked_lm_logits).requires_grad_(True)
    
        # 初始化优化器参数
        optim = torch.optim.Adam([epsilon], lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=10, gamma=0.9)
    
        # 获取原始词的编码和词嵌入
        original_word = current_text.words[index]
        print(f"Original word: {original_word}")
        original_ids = self._lm_tokenizer.encode(original_word, add_special_tokens=False)
        print(f"Original ids: {original_ids}")
        # 确保使用正确的数据类型并移动到GPU
        original_tensor = torch.tensor(original_ids, dtype=torch.long, device='cuda')
        print(f"Original tensor shape: {original_tensor.shape}")
        print(f"Original tensor: {original_tensor}")
        # 使用 no_grad 来优化内存使用，并确保模型和张量的精度匹配
        with torch.no_grad():
            
            original_embedding = self._language_model.get_input_embeddings()(original_tensor)
            if len(original_embedding.shape) > 1:
                original_embedding = original_embedding.mean(dim=0)
    
        best_loss = float('inf')
        
        # 优化循环
        found_valid_replacement = False
        iter = 0
        while not found_valid_replacement and iter < self.num_iterations:
            optim.zero_grad()
        
            # 计算当前logits并应用温度缩放
            current_logits = (masked_lm_logits + epsilon) / max(self.temperature, 1e-2)
            current_logits = torch.clamp(current_logits, min=-1e2, max=1e2)
            current_probs = torch.nn.functional.softmax(current_logits, dim=-1).detach()
            current_probs.requires_grad = True
            
            # 计算流畅度损失
            flu_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(current_logits, dim=-1),
                torch.nn.functional.softmax(masked_lm_logits / self.temperature, dim=-1),
                reduction='batchmean'
            )
            
            # 计算与原词的语义相似度损失
            candidate_embeddings = self._language_model.get_input_embeddings()(torch.arange(self._lm_tokenizer.vocab_size).to(self.device))
            # 使用候选词嵌入将 current_probs 转换为嵌入空间
            current_embeddings = torch.matmul(current_probs, candidate_embeddings)
            # 扩展原词嵌入
            original_embedding_expanded = original_embedding.unsqueeze(0).expand(current_probs.size(0), -1)
            print("current_probs shape:", current_probs.shape)
            print("original_embedding shape:", original_embedding.shape)
            print("original_embedding_expanded shape:", original_embedding_expanded.shape)
            diff_loss = 1 - torch.cosine_similarity(current_embeddings, original_embedding_expanded, dim=-1).mean()
            
            # 总损失
            loss = flu_loss + self.diff_weight * diff_loss
            
            if loss < best_loss:
                best_loss = loss.item()
                top_values, top_indices = torch.topk(current_probs, k=self.max_candidates)
                
                for row_indices, row_values in zip(top_indices, top_values):
                    for idx, prob in zip(row_indices, row_values):
                        word = self._lm_tokenizer.decode(idx)
                        if (prob.item() >= self.min_confidence 
                            and utils.is_one_word(word)
                            and not utils.check_if_punctuations(word)
                            and word != original_word
                            and word.isalpha()):
                            best_replacements.append(word)
                            found_valid_replacement = True
        
            loss.backward()
            optim.step()
            scheduler.step()
        
            # 添加噪声增加多样性
            noise = torch.normal(0, self.noise_std, size=epsilon.size(), device=epsilon.device)
            epsilon.data += noise
    
            if not found_valid_replacement:
                self.temperature = min(self.temperature * 1.1, self.max_temperature)
                iter += 1
                if iter >= self.max_iterations:
                    print("Reached maximum iterations with no valid replacement found.")
                    break
    
        # 根据语义相似度对候选词进行排序
        print("best_replacements:", best_replacements)
        scored_replacements = self._rank_and_filter_word_replacements(
            best_replacements, 
            original_embedding
            )
        print("scored_replacements:", scored_replacements)
        
        # 返回排序后的替换词（不包含分数）
        return [word for word, _ in scored_replacements[:self.max_candidates]]

    def _rank_and_filter_word_replacements(
            self, 
            best_replacements, 
            original_embedding, 
            top_n=10, 
            similarity_weight=0.8,  # 增加相似度权重
            frequency_weight=0.2    # 降低频率权重
        ):
            """
            对替换词进行排序和过滤
            """
            from collections import Counter
            
            # 去重并计数
            replacement_counts = Counter(best_replacements)
            max_count = max(replacement_counts.values())  # 获取最大频率
            
            # 按出现频率和语义相似度排序
            scored_replacements = []
            unique_replacements = set()
            
            # 过滤掉常见的停用词
            stop_words = {'en', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with'}
        
            for replacement in replacement_counts.most_common():
                word = replacement[0]
                
                # 跳过停用词
                if word.lower() in stop_words:
                    continue
                
                # 跳过已处理的词
                if word in unique_replacements:
                    continue
                
                # 将元组转换为字符串
                if isinstance(word, tuple):
                    word = "".join(word)
                
                # 获取词的嵌入
                replacement_ids = self._lm_tokenizer.encode(word, add_special_tokens=False)
                replacement_embedding = self._language_model.get_input_embeddings()(
                    torch.tensor(replacement_ids, device=self.device)
                ).mean(dim=0)
                
                # 计算与原始嵌入的相似度
                similarity = torch.cosine_similarity(original_embedding, replacement_embedding, dim=0).item()
                
                # 归一化频率分数
                frequency_score = replacement_counts[word] / max_count
                
                # 计算综合得分，添加长度惩罚项
                length_penalty = 1.0 if len(word) > 2 else 0.5  # 惩罚过短的词
                
                combined_score = (
                    similarity_weight * similarity + 
                    frequency_weight * frequency_score
                ) * length_penalty
                
                # 添加到结果列表
                scored_replacements.append((word, combined_score))
                unique_replacements.add(word)
        
            # 按综合得分降序排序
            scored_replacements.sort(key=lambda x: x[1], reverse=True)
        
            # 返回top N个
            return scored_replacements[:top_n]

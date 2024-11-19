from textattack.search_methods import SearchMethod
import spacy
import numpy as np
import torch
from torch.nn.functional import softmax

class MyCustomSearchMethod(SearchMethod):
    """A custom search method for attacking text models."""

    def __init__(self, wir_method="unk", unk_token="[UNK]"):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.nlp = spacy.load("en_core_web_sm")  # Load the spaCy model



    def _get_index_order(self, initial_text, max_len=-1):
        """Custom logic to return word indices of `initial_text` in descending order of importance."""

        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        # Identify phrases to replace
        phrases = []
        for chunk in self.nlp(initial_text.text).noun_chunks:
            phrases.append((chunk.start, chunk.end, "noun-phrase"))
        for token in self.nlp(initial_text.text):
            if token.pos_ == "VERB":
                phrases.append((token.i, token.i + 1, "verb-phrase"))
            elif token.dep_ == "fixed":
                phrases.append((token.i, token.i + 1, "fixed-expression"))

        if self.wir_method == "unk":
            leave_one_texts = [initial_text.replace_words_at_indices(range(start, end), [self.unk_token] * (end - start)) for start, end, _ in phrases]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            leave_one_texts = [initial_text.replace_words_at_indices(range(start, end), [self.unk_token] * (end - start)) for start, end, _ in phrases]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])
            softmax_saliency_scores = softmax(torch.Tensor(saliency_scores), dim=0).numpy()

            delta_ps = []
            for idx, (start, end, _) in enumerate(phrases):
                if search_over:
                    delta_ps = delta_ps + [0.0] * (len(softmax_saliency_scores) - len(delta_ps))
                    break

                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=list(range(start, end)),
                )
                if not transformed_text_candidates:
                    delta_ps.append(0.0)
                    continue
                swap_results, search_over = self.get_goal_results(transformed_text_candidates)
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [initial_text.delete_word_at_index(i) for i, _, _ in phrases]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(len_text)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, (start, end, _) in enumerate(phrases):
                matched_tokens = [word2token_mapping[idx] for idx in range(start, end)]
                matched_tokens = [token for sublist in matched_tokens for token in sublist]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")

        if self.wir_method != "random":
            index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            search_over = False

        return index_order, search_over

    def perform_search(self, initial_text):
        """Perform the search using the custom method."""
        # Implement your custom search logic here
        index_order, search_over = self._get_index_order(initial_text)
        # Example logic: simply return the initial text
        #输出重要性排序
        print(f"Index Order: {index_order}")
        
        return initial_text

# Example usage
if __name__ == "__main__":
    search_method = MyCustomSearchMethod()
    # Example usage of the search method
    # Note: This is a placeholder, you would need to create an initial_text object
    # that is compatible with the search method
    # initial_text = ...
    # result = search_method.perform_search(initial_text)
    # print(result)
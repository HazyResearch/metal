import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from metal.mmtl.utils.dataloaders import add_labels_to_payload


def get_bert_spacy_index(spc, tokenizer, add_cls):
    """
    Map spacy to bert
    """
    bert_tokens = []
    orig_to_tok_map = []
    if add_cls:
        bert_tokens.append("[CLS]")
    for orig_token in spc:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token.text))
    bert_tokens.append("[SEP]")
    return orig_to_tok_map, bert_tokens


# Function to add BLEU labels
def add_bleu_labels(payload):
    """
    Adds 1-gram bleu score labelset for sentence similarity tasks
    """

    def get_bleu_label(it):
        # toks, segs = payload.data_loader.dataset[it][0]
        toks = payload.data_loader.dataset.bert_tokens[it]
        toks = payload.data_loader.dataset.bert_tokenizer.convert_ids_to_tokens(toks)
        segs = payload.data_loader.dataset.bert_segments[it]
        toks, segs = np.array(toks), np.array(segs)
        sent1 = list(toks[segs == 0])
        sent2 = list(toks[segs == 1])
        bleu_score = sentence_bleu(sent1, sent2, weights=(1, 0, 0, 0))
        return bleu_score

    add_labels_to_payload(payload, "BLEU", get_bleu_label)


# Adding NER labels from Spacy
def add_spacy_ner_labels(payload):
    """
    Adds spacy ner labelset, maps Spacy to Wordpiece tokens
    """

    def get_spacy_ner_tags(it):
        sent1, sent2 = payload.data_loader.dataset.nlp_spacy[it]
        tokenizer = payload.data_loader.dataset.bert_tokenizer
        sent1_token_map, sent1_bert = get_bert_spacy_index(
            sent1, tokenizer, add_cls=True
        )
        sent2_token_map, sent2_bert = get_bert_spacy_index(
            sent2, tokenizer, add_cls=False
        )

        spacy_ner_tags = np.zeros((len(sent1_bert + sent2_bert),))

        # This condition should be true if we've done things correctly!
        bert_tokens_orig = payload.data_loader.dataset.bert_tokenizer.convert_ids_to_tokens(
            payload.data_loader.dataset.bert_tokens[it]
        )
        assert np.array_equal(sent1_bert + sent2_bert, bert_tokens_orig)

        # Creating tags -- string labels for now!
        sent_1_tags = np.zeros((len(sent1_bert),)).astype(str)
        sent_2_tags = np.zeros((len(sent2_bert),)).astype(str)

        # label_ for string, label for int
        for ent in sent1.ents:
            sent_1_tags[
                sent1_token_map[ent.start] : sent1_token_map[ent.end]
            ] = ent.label_
        for ent in sent2.ents:
            sent_2_tags[
                sent2_token_map[ent.start] : sent2_token_map[ent.end]
            ] = ent.label_

        spacy_ner_tags = list(sent_1_tags) + list(sent_2_tags)
        return spacy_ner_tags

    add_labels_to_payload(payload, "SPACY_NER", get_spacy_ner_tags)


auxiliary_task_functions = {"BLEU": add_bleu_labels, "SPACY_NER": add_spacy_ner_labels}

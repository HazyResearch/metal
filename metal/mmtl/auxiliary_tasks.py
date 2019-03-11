import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from metal.mmtl.utils.dataloaders import add_labels_to_payload

SPACY_INFO = {
    # 0.0 is null, for clarity
    "NER_TAGS": [
        "0.0",
        "PERSON",
        "NORP",
        "FAC",
        "ORG",
        "GPE",
        "LOC",
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "DATE",
        "TIME",
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
    ],
    "POS_TAGS": [
         "0.0",
         "PUNCT",
         "SYM",
         "X",
         "ADJ",
         "VERB",
         "CONJ",
         "CCONJ",
         "NUM",
         "DET",
         "ADV",
         "ADP",
         "", 
         "NOUN",
         "PROPN",
         "PART",
         "PRON",
         "SPACE",
         "INTJ",
    ],
}


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

def get_bert_spacy_index_char(spc, bert, add_cls):
    """
    Map spacy to bert
    """
    orig_to_tok_map = []
    bert_tokens = []
    if add_cls:
        bert_tokens.append("[CLS]")
   # Removing hashes
    bert_no_hash = [t if '##' not in t else t[2:] for t in bert]
    for orig_token in spc:
        orig_to_tok_map.append(len(bert_tokens))
        bert_end = bert_no_hash[len(bert_tokens):]
        # NOTE: Using all lower for now!
        orig_token = orig_token.text.lower()
        for t in bert_end:
            if t.lower() in orig_token:
                orig_token = orig_token[len(t):]
                bert_tokens.append(t)
            elif orig_token in t.lower():
                bert_tokens.append(t)
            if (t.lower() not in orig_token) and (orig_token not in t.lower()):
                break
            orig_token = orig_token[len(t):]
            bert_tokens.append(t)
        
    bert_tokens.append("[SEP]")
    if len(bert_tokens) != len(bert):
        import ipdb; ipdb.set_trace()
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
        return float(bleu_score)

    return add_labels_to_payload(payload, "BLEU", label_fn=get_bleu_label)


# Function add THIRD labels
def add_third_labels(payload):
    """Marks whether each unmasked token is in the 1st/2nd/3rd third of the sentence"""

    def mark_thirds(x):
        tokens = x
        count = len(tokens)
        Y_thirds = np.ceil(
            np.array([idx * 3 / count for idx in range(1, count + 1)])
        ).astype(np.int64)
        return Y_thirds

    X = payload.data_loader.dataset.bert_tokens
    Y = []
    for x in X:
        Y.append(mark_thirds(x))

    return add_labels_to_payload(payload, "THIRD", label_set=Y)


# Adding NER labels from Spacy
def add_spacy_ner_labels(payload):
    """
    Adds spacy ner labelset, maps Spacy to Wordpiece tokens
    """

    def get_spacy_ner_tags(it):
        sent1, sent2 = payload.data_loader.dataset.spacy_tokens[it]
        tokenizer = payload.data_loader.dataset.bert_tokenizer
        sent1_token_map, sent1_bert = get_bert_spacy_index(
            sent1, tokenizer, add_cls=True
        )
        sent2_token_map, sent2_bert = get_bert_spacy_index(
            sent2, tokenizer, add_cls=False
        )

        spacy_ner_tags = np.zeros((len(sent1_bert + sent2_bert),))

        # This condition should be true if we've done things correctly!
        bert_tokenizer = payload.data_loader.dataset.bert_tokenizer
        bert_tokens_orig = bert_tokenizer.convert_ids_to_tokens(
            payload.data_loader.dataset.bert_tokens[it]
        )
        #        if not np.array_equal(sent1_bert + sent2_bert, bert_tokens_orig):
        #            import ipdb; ipdb.set_trace()

        # Sometimes, weird things happen with apostrophes; just make sure length aligns
        # assert len(sent1_bert + sent2_bert) == len(bert_tokens_orig)

        # Creating tags -- string labels for now!
        sent_1_tag_strs = np.zeros((len(sent1_bert),)).astype(str)
        sent_2_tag_strs = np.zeros((len(sent2_bert),)).astype(str)

        # label_ for string, label for int
        # HACK: due to occasional misalignment, need to make sure we don't go over end of sentence
        for ent in sent1.ents:
            sent_1_tag_strs[
                sent1_token_map[ent.start] : sent1_token_map[
                    min(ent.end, len(sent1_token_map) - 1)
                ]
            ] = ent.label_
        for ent in sent2.ents:
            sent_2_tag_strs[
                sent2_token_map[ent.start] : sent2_token_map[
                    min(ent.end, len(sent2_token_map) - 1)
                ]
            ] = ent.label_

        sent_1_tags = [SPACY_INFO["NER_TAGS"].index(tag) + 1 for tag in sent_1_tag_strs]
        sent_2_tags = [SPACY_INFO["NER_TAGS"].index(tag) + 1 for tag in sent_2_tag_strs]

        spacy_ner_tags = list(sent_1_tags) + list(sent_2_tags)

        # HACK: Dealing with misalignment by padding/truncating
        while len(spacy_ner_tags) < len(bert_tokens_orig):
            # Because of 1-indexing for metal labels!
            spacy_ner_tags.append(1)

        if len(spacy_ner_tags) > len(bert_tokens_orig):
            spacy_ner_tags = spacy_ner_tags[: len(bert_tokens_orig)]

        assert len(spacy_ner_tags) == len(bert_tokens_orig)
        return spacy_ner_tags

    return add_labels_to_payload(payload, "SPACY_NER", label_fn=get_spacy_ner_tags)

# Adding NER labels from Spacy
def add_spacy_pos_labels(payload):
    """
    Adds spacy ner labelset, maps Spacy to Wordpiece tokens
    """

    def get_spacy_pos_tags(it):
        sent1, sent2 = payload.data_loader.dataset.spacy_tokens[it]
        tokenizer = payload.data_loader.dataset.bert_tokenizer
        #sent1_token_map, sent1_bert = get_bert_spacy_index(
        #    sent1, tokenizer, add_cls=True
        #)
        #sent2_token_map, sent2_bert = get_bert_spacy_index(
        #    sent2, tokenizer, add_cls=False
        #)

        bert_tokenizer = payload.data_loader.dataset.bert_tokenizer
        bert_tokens_orig = bert_tokenizer.convert_ids_to_tokens(
            payload.data_loader.dataset.bert_tokens[it])
        segments_orig = np.array(payload.data_loader.dataset.bert_segments[it])
        bert_tokens_arr, segments_arr = np.array(bert_tokens_orig), np.array(segments_orig)
        sent_1_bert_orig = list(bert_tokens_arr[segments_arr==0])
        sent_2_bert_orig = list(bert_tokens_arr[segments_arr==1])
        sent1_token_map, sent1_bert = get_bert_spacy_index_char(
            sent1, sent_1_bert_orig, add_cls=True)

        sent2_token_map, sent2_bert = get_bert_spacy_index_char(
            sent2, sent_2_bert_orig, add_cls=False)

        spacy_pos_tags = np.zeros((len(sent1_bert + sent2_bert),))

        # This condition should be true if we've done things correctly!
        
        # Sometimes, weird things happen with apostrophes; just make sure length aligns
        # assert len(sent1_bert + sent2_bert) == len(bert_tokens_orig)

        # Creating tags -- string labels for now!
        sent_1_tag_strs = np.zeros((len(sent1_bert),)).astype(str)
        sent_2_tag_strs = np.zeros((len(sent2_bert),)).astype(str)

        # label_ for string, label for int
        # HACK: due to occasional misalignment, need to make sure we don't go over end of sentence
        for ii, token in enumerate(sent1):
            sent_1_tag_strs[
                sent1_token_map[ii] : sent1_token_map[
                    min(ii+1, len(sent1_token_map) - 1)
                ]
            ] = token.pos_
        for ii, token in enumerate(sent2):
            sent_2_tag_strs[
                sent2_token_map[ii] : sent2_token_map[
                    min(ii+1, len(sent2_token_map) - 1)
                ]
            ] = token.pos_

        sent_1_tags = [SPACY_INFO["POS_TAGS"].index(tag) + 1 for tag in sent_1_tag_strs]
        sent_2_tags = [SPACY_INFO["POS_TAGS"].index(tag) + 1 for tag in sent_2_tag_strs]

        spacy_pos_tags = list(sent_1_tags) + list(sent_2_tags)

        # HACK: Dealing with misalignment by padding/truncating
        while len(spacy_pos_tags) < len(bert_tokens_orig):
            # Because of 1-indexing for metal labels!
            import ipdb; ipdb.set_trace()
            spacy_pos_tags.append(1)
            print('Misaligned POS tags!')

        if len(spacy_pos_tags) > len(bert_tokens_orig):
            spacy_pos_tags = spacy_pos_tags[: len(bert_tokens_orig)]
            print('Misaligned POS tags!')

        assert len(spacy_pos_tags) == len(bert_tokens_orig)
        return spacy_pos_tags

    return add_labels_to_payload(payload, "SPACY_POS", label_fn=get_spacy_pos_tags)

auxiliary_task_functions = {
    "BLEU": add_bleu_labels,
    "THIRD": add_third_labels,
    "SPACY_NER": add_spacy_ner_labels,
    "SPACY_POS": add_spacy_pos_labels,
}

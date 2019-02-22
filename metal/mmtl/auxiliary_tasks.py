# import numpy as np

# from metal.mmtl.utils.dataset_utils import get_dataloader_with_label

# from nltk.translate.bleu_score import sentence_bleu


# Function to create BLEU dataloaders
# def get_bleu_dataloader(dataloader):
#     def get_bleu_label(it):
#         toks, segs = it[0]
#         toks = dataloader.dataset.tokenizer.convert_ids_to_tokens(toks)
#         toks, segs = np.array(toks), np.array(segs)
#         sent1 = list(toks[segs == 0])
#         sent2 = list(toks[segs == 1])
#         bleu_score = sentence_bleu(sent1, sent2, weights=(1, 0, 0, 0))
#         return bleu_score

#     return get_dataloader_with_label(dataloader, get_bleu_label)

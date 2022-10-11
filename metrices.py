from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import numpy as np


def bleu(references,hypothesis):
    splited_references = []
    splited_hypothesises = []
    for reference in references:
        splited_reference = reference.split(' ')
        splited_references.append(splited_reference)
    for hypothesis in hypothesis:
        splited_hypothesis = hypothesis.split(' ')
        splited_hypothesises.append(splited_hypothesis)

    score = corpus_bleu(splited_references,splited_hypothesises,smoothing_function = SmoothingFunction().method1)
    return score


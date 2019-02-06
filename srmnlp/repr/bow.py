""" Bag of Words

"""
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="corpus file (text)", default='data/asimov-tiny.txt')
args = parser.parse_args()


if __name__ == '__main__':

  bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
      token_pattern=r'\b\w+\b',  # what is going on here?
      min_df=1)

  if args.input == 'numbers':
    from generate_synthetic import generate_numbers
    sentences = generate_numbers()
  else:
    sentences = sent_tokenize(open(args.input).read())

  analyze = bigram_vectorizer.build_analyzer()
  print(analyze(sentences[0]))

  # fit on data
  bigram_vectorizer.fit(sentences)

  # tranform to vectors
  vectors = bigram_vectorizer.transform(sentences)

  # TF-IDF transformer instance
  tf_transformer = TfidfTransformer()
  tf_transformer.fit(vectors)

  # sentence = ['the last question']
  sentence = [sentences[0]]
  print(sentence)
  print(tf_transformer.transform(
    bigram_vectorizer.transform(sentence)
    ).toarray())

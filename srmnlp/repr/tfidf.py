from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="corpus file (text)", default='data/asimov.txt')
args = parser.parse_args()


if __name__ == '__main__':

  # read from file; split sentences
  sentences = sent_tokenize(open(args.input).read())
  # create a vectorizer instance
  vectorizer = CountVectorizer()
  # fit on data
  vectorizer.fit(sentences)
  # transform sentences
  vectors = vectorizer.transform(sentences)

  # Frequency-based transformer
  tf_transformer = TfidfTransformer()
  tf_transformer.fit(vectors)

  print(sentences[0])
  print(tf_transformer.transform(
    vectorizer.transform([sentences[0]])
    ).toarray())

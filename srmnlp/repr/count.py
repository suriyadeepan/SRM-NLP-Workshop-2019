from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="corpus file (text)", default='data/asimov.txt')
args = parser.parse_args()


if __name__ == '__main__':

  # create an instance of sklearn's Count Vectorizer
  vectorizer = CountVectorizer()
  sentences = sent_tokenize(open(args.input).read())
  vectorizer.fit(sentences)

  print(sentences[0])
  print(vectorizer.transform([sentences[0]]).toarray())

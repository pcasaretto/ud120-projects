#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import re

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### project part 2: comment out the line below
        words = text_string

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        words = []
        for word in text_string.split():
            stemmer = SnowballStemmer("english")
            stemWord = stemmer.stem(word)
            words.append(stemWord)
        # stemmer = SnowballStemmer("english")
        # ar_words = map(lambda x: stemmer.stem(x.strip()), text_string.split(' '))
        # ar_words = filter(lambda x: x != '', ar_words)
        # words = " ".join(ar_words)
        # words = re.sub(r'\s+', ' ', words)
    return string.join(words)


def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

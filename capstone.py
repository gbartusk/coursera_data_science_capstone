import os
import sys
import time
import re
import string
from unidecode import unidecode
import csv
import math

# - natural language processing libraries
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.util import ngrams

# - links:
#   > http://www.nltk.org/book_1ed/
#   > http://nlp.stanford.edu/IR-book/html/htmledition/contents-1.html

def main():
    """
    """
    # - define input file paths
    dir_root = "/Users/Gus/Google Drive/dev/coursera/data_sciences/10_capstone"
    dir_in = os.path.join(dir_root, "sample_data/en_US")
    #dir_in = os.path.join(dir_root, "final/en_US/")
    file_size = ''  # files of size 10 and 40 saved in sample data
    fp_profanity = os.path.join(dir_root, "profanity_list_en.txt")
    fp_twitter = os.path.join(dir_in, "en_US.twitter"+file_size+".txt")
    fp_news = os.path.join(dir_in, "en_US.news"+file_size+".txt")
    fp_blogs = os.path.join(dir_in, "en_US.blogs"+file_size+".txt")

    # - define output file paths
    #dir_out = os.path.join(dir_root, "data_cache/en_US")
    dir_out = os.path.join(dir_root, "data_cache/en_US/test")
    fp_out_twitter = os.path.join(dir_out, "twitter_sent_tokenize.csv")
    fp_out_news = os.path.join(dir_out, "news_sent_tokenize.csv")
    fp_out_blogs = os.path.join(dir_out, "blogs_sent_tokenize.csv")
    fp_out_1gram = os.path.join(dir_out, "1grams.csv")
    fp_out_2gram = os.path.join(dir_out, "2grams.csv")
    fp_out_3gram = os.path.join(dir_out, "3grams.csv")
    fp_out_4gram = os.path.join(dir_out, "4grams.csv")
    fp_out_5gram = os.path.join(dir_out, "5grams.csv")

    # - define args
    run_args = [
        {"n":1, "fp_out": fp_out_1gram},
        {"n":2, "fp_out": fp_out_2gram},
        {"n":3, "fp_out": fp_out_3gram},
        {"n":4, "fp_out": fp_out_4gram},
        {"n":5, "fp_out": fp_out_5gram},
    ]
    # - read in profanity set
    dat_profanity = set(line.strip() for line in open(fp_profanity))
    
    # - return data
    ngram_freq = {}
    
    # - process all
    ttl_start_time = time.time()
    for run_cv in run_args:
        print("running ngram: " + str(run_cv["n"]))
        start_time = time.time()
        process_file(ngram_freq, fp_twitter, dat_profanity, n=run_cv["n"], log=True)
        process_file(ngram_freq, fp_news, dat_profanity, n=run_cv["n"], log=True)
        process_file(ngram_freq, fp_blogs, dat_profanity, n=run_cv["n"], log=True)
        write_dict_freq(run_cv["fp_out"], ngram_freq, log=True, min_freq=1)
        end_time = time.time()
        run_time = end_time - start_time
        print("run time: " + str(round(run_time/60,2)) + " min")
        print("~"*50)
    ttl_end_time = time.time()
    ttl_run_time = ttl_end_time - ttl_start_time
    print("total run time: " + str(round(ttl_run_time/60,2)) + " min")
    print("~"*50)

    # - process files
    # start_time = time.time()
    # process_file(ngram_freq, fp_twitter, dat_profanity, n=2, log=True)
    # process_file(ngram_freq, fp_news, dat_profanity, n=2, log=True)
    # process_file(ngram_freq, fp_blogs, dat_profanity, n=2, log=True)
    # write_dict_freq(fp_out_2gram, ngram_freq, log=True)
    # end_time = time.time()
    # run_time = end_time - start_time
    # print("total run time: " + str(round(run_time/60,2)) + " min")


def process_file(ngram_freq, file_in, profanity, n, log=False):
    """
    """
    # - log: file
    if log: 
        print("processing (ngram=" + str(n) + "): " + os.path.basename(file_in))
        start_time = time.time()

    # - precompile regular expressions and translation tables
    re_num = re.compile(r'\d+[a-z]*')
    re_whitespace = re.compile(r'\s+')
    re_dash = re.compile(r'-')
    ttbl_punc = {ord(c): None for c in string.punctuation}

    # - open input files
    with open(file_in, mode='r', encoding='utf-8') as f_in:
        # - loop over lines in file
        line_count = 1
        for line_cv in f_in:
            # - log: rows processed
            if log and line_count % 250000 == 0: print("  > processed: " + "{:,}".format(line_count) + " lines")
            # - stip the newline character and sentence tokenize
            line_sentences = sent_tokenize(line_cv.strip(), language='english')
            # - write to output file
            for sent_cv in line_sentences:
                # - clean sentence
                sent_clean = sent_cv.lower()                        # lowercase
                sent_clean = unidecode(sent_clean)                  # convert to ascii
                sent_clean = re_dash.sub(' ', sent_clean)           # replace dash with space
                sent_clean = sent_clean.translate(ttbl_punc)        # remove punctuation
                sent_clean = re_num.sub('NUM', sent_clean)          # replace numbers with a marker
                # remove null characters (in twitter)
                #sent_clean = re_whitespace.sub(' ', sent_clean)    # not need since tokenizer splits on \w+
                # - word tokenize and remove profanity
                sent_tkns = [w if w not in profanity else 'PROFANITY' for w in word_tokenize(sent_clean)]
                # - add ngrams to frequency dictionary
                for ng in ngrams(sent_tkns,n):
                    ng_str = ' '.join(ng)
                    ngram_freq[ng_str] = ngram_freq.get(ng_str,0)+1
                # - debug
                # print(str(line_count) + ": " + sent_cv + "\n  => " + sent_clean)
                # print("  => " + str(list(ngrams(sent_tkns,n))))
                # print("~"*20)
            line_count += 1

    # - log: run time
    if log: 
        end_time = time.time()
        run_time = end_time - start_time
        print("  > run time: " + str(round(run_time/60,2)) + " min")


def write_dict_freq(file_out, ngram_freq, log=False, min_freq=1):
    """
    """
    with open(file_out, mode='w', encoding='utf-8') as f_out:
        csv_writer = csv.writer(f_out, quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writerow(('ngram','freq'))
        for row in ngram_freq.items():
            # - only write if row has frequency greater than threshold
            if row[1] >= min_freq: csv_writer.writerow(row)
    if log: print("out file: " + file_out)


def clean_sentence(sent):
    """
    NOT IN USE!
    """
    # - convert to lower case and remove punctuation
    sent_clean = sent.lower().translate(None, string.punctuation)
    return sent_clean


def testing():
    # - tokenize on sentence and word
    ex_txt = "hello there Mr. Bartuska, How are you? The weather is great and I enjoy Python. cheers!"
    print(sent_tokenize(ex_txt))
    print(word_tokenize(ex_txt, language='english'))

    # - stop words (pre-defined by nltk)
    stop_words = set(stopwords.words('english'))
    print(stop_words)
    words = word_tokenize(ex_txt)
    print(words)
    filtered_sent = []
    for w in words:
        if w not in stop_words:
            filtered_sent.append(w)
    print(filtered_sent)
    filtered_sent = [w for w in words if not w in stop_words]
    print(filtered_sent)

    # - stemming
    ps = PorterStemmer()
    example_words = [python,pythoner,pythoning,pythoned,pythonly]
    # for w in example_words:
    #     print(ps.stem(w))
    new_text = "it is very important to be pothonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
    words = word_tokenize(new_text)
    for w in words:
        print(ps.stem(w))


if __name__ == "__main__": 
    main()
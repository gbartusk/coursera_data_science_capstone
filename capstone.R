#
# - Coursera Data Science Capstone
#

# - libraries
library(rJava)          # - connection to java (openNLP and RWeka both depend on java)
library(NLP)            # - nlp infrastructure (widely used by other pkgs)
library(openNLP)        # - interface to apache oepnNLP library (in java)
library(RWeka)          # - Weka data mining software (in java), useful for ngrams
library(qdap)           # - fxns for qualitative discourse analysis
library(magrittr)       # - coding semantics, piping
library(tm)             # - comprehensive text mining framework
library(SnowballC)      # - stemmers (required for tm stemming)
library(tau)            # - text analysis utilities
library(RTextTools)     # - 
library(wordnet)        # - 
library(ggplot2)        # - general plotting
library(Rgraphviz)      # - plot network graphs of correlation between words
library(wordcloud)      # - word clouds
library(dplyr)          # - data processing
library(stringr)        # - string manipulation

# - Notes:
#   > https://www.ibm.com/developerworks/community/blogs/nlp/entry/tokenization?lang=en
#   > http://nlp.stanford.edu/IR-book/html/htmledition/contents-1.html
#   > https://rpubs.com/lmullen/nlp-chapter
#   > http://onepager.togaware.com/TextMiningO.pdf
#   > http://cran.r-project.org/web/packages/tm/tm.pdf
#   > http://tm.r-forge.r-project.org/index.html
#   >
#   >


# - quiz: week 1
{    
    # - 1: The en_US.blogs.txt file is how many megabytes?
    #   http://stackoverflow.com/questions/2365100/converting-bytes-to-megabytes
    file.size(file_us_blog) / 1000^2
    pryr::object_size(dat_us_blog)
    
    # - 2: The en_US.twitter.txt has how many lines of text
    length(dat_us_twit)
    
    # - 3: What is the length of the longest line seen in any of the three 
    #   en_US data sets?
    # - > awk '{ if (length($0) > max) {max = length($0); maxline = $0} } END { print max }' en_US/en_US.blogs.txt
    max(nchar(dat_us_blog))
    max(nchar(dat_us_twit))
    max(nchar(dat_us_news))
    
    # - 4: In the en_US twitter data set, if you divide the number of lines 
    #   where the word "love" (all lowercase) occurs by the number of lines 
    #   the word "hate" (all lowercase) occurs, about what do you get?
    # - > awk '{ if ($0 ~ /love/) {max++} } END { print max }' en_US/en_US.twitter.txt
    # - > awk '{ if ($0 ~ /hate/) {max++} } END { print max }' en_US/en_US.twitter.txt
    twit_love <- sum(grepl("love",dat_us_twit))
    twit_hate <- sum(grepl("hate",dat_us_twit))
    twit_love / twit_hate
    
    # - 5: The one tweet in the en_US twitter data set that matches the word 
    #   "biostats" says what?
    grep("biostats", dat_us_twit, value = TRUE)
    
    # - 6: How many tweets have the exact characters 
    #   "A computer once beat me at chess, but it was no match for me at kickboxing". 
    #   (I.e. the line matches those characters exactly.)
    # - > awk '{ if ($0 ~ /^A computer once beat me at chess, but it was no match for me at kickboxing/) {print $0; ttl++} } END { print ttl }' en_US/en_US.twitter.txt
    sum(grepl("^A computer once beat me at chess, but it was no match for me at kickboxing$", dat_us_twit))
    which(dat_us_twit == "A computer once beat me at chess, but it was no match for me at kickboxing")
    
}
# - quiz: week 3
{
    # - raw and 2 processed corpuses: 184 MB
    # - unigrams: 24 MB
    # - bigrams w/out stop words: 424 MB
    # - trigrams w/ and w/out stop words: 1.42 GB
    # - four-grams w/ and w/out stop words: 1.8 GB
    # - Total: ~4GB+
    
    # - directories
    setwd("/Users/Gus/Google Drive/dev/coursera/data_sciences/10_capstone")
    sample_dir <- file.path("sample_data", "en_US")
    data_dir <- file.path("final", "en_US")
    data_cache_dir <- file.path("data_cache","en_US")
    
    # - load all corpus
    corpus_all_raw <- tm::Corpus(tm::DirSource(sample_dir, encoding="UTF-8"))
    corpus_all <- clean_corpus(corpus_all_raw)
    corpus_all_stops <- clean_corpus(corpus_all_raw, rm_stop_words=FALSE)
    corpus_all_stops_stem <- clean_corpus(corpus_all_raw, rm_stop_words=FALSE, stem=TRUE)
    
    # - define n-gram tokenizers
    bigram_tokenizer <- function(x) unlist(
        lapply(NLP::ngrams(NLP::words(x), 2), paste, collapse = " "), use.names = FALSE)
    trigram_tokenizer <- function(x) unlist(
        lapply(NLP::ngrams(NLP::words(x), 3), paste, collapse = " "), use.names = FALSE)
    fourgram_tokenizer <- function(x) unlist(
        lapply(NLP::ngrams(NLP::words(x), 4), paste, collapse = " "), use.names = FALSE)
    fivegram_tokenizer <- function(x) unlist(
        lapply(NLP::ngrams(NLP::words(x), 5), paste, collapse = " "), use.names = FALSE)
    
    # - filepaths rds
    path_df_all_freq_uni <- file.path(data_cache_dir, "1gram_freq_noStops_noStem.rds")
    path_df_all_freq_bi <- file.path(data_cache_dir, "2gram_freq_noStops_noStem.rds")
    path_df_all_freq_stops_bi <- file.path(data_cache_dir, "2gram_freq_Stops_noStem.rds")
    path_df_all_freq_stops_tri <- file.path(data_cache_dir, "3gram_freq_stops_noStem.rds")
    path_df_all_freq_stops_stem_tri <- file.path(data_cache_dir, "3gram_freq_stops_stem.rds")
    path_df_all_freq_stops_four <- file.path(data_cache_dir, "4gram_freq_stops_noStem.rds")
    path_df_all_freq_stops_stem_four <- file.path(data_cache_dir, "4gram_freq_stops_stem.rds")
    path_df_all_freq_stops_stem_five <- file.path(data_cache_dir, "5gram_freq_stops_stem.rds")
    
    # - unigram tokenization: stop words removed
    tdm_all_uni <- tm::TermDocumentMatrix(corpus_all)
    df_all_freq_uni <- get_freq_df(tdm_all_uni)
    # - cache
    saveRDS(df_all_freq_uni, path_df_all_freq_uni)
    rm(tdm_all_uni); gc()
    # - load rds
    #df_all_freq_uni <- readRDS(path_df_all_freq_uni)
    
    
    # - bigram tokenization: with stop words
    tdm_all_stops_bi <- tm::TermDocumentMatrix(corpus_all_stops, 
        control = list(tokenize = bigram_tokenizer))
    df_all_freq_stops_bi <- get_freq_df(tdm_all_stops_bi)
    saveRDS(df_all_freq_stops_bi, path_df_all_freq_stops_bi)
    rm(tdm_all_stops_bi); gc()
    # - bigram tokenization: stop words removed
    tdm_all_bi <- tm::TermDocumentMatrix(corpus_all, 
        control = list(tokenize = bigram_tokenizer))
    df_all_freq_bi <- get_freq_df(tdm_all_bi)
    # - cache
    saveRDS(df_all_freq_bi, path_df_all_freq_bi)
    rm(tdm_all_bi); gc()
    # - load rds
    #df_all_freq_bi <- readRDS(path_df_all_freq_bi)
    
    # - trigram tokenization, w/ stop words
    tdm_all_tri_stops <- tm::TermDocumentMatrix(corpus_all_stops, 
        control = list(tokenize = trigram_tokenizer))
    df_all_freq_stops_tri <- get_freq_df(tdm_all_tri_stops)
    # - cache 3-gram w/ stops
    saveRDS(df_all_freq_stops_tri, path_df_all_freq_stops_tri)
    rm(tdm_all_tri_stops); gc()
    # - trigram tokenization: w/ stops + stemming
    tdm_all_tri_stops_stem <- tm::TermDocumentMatrix(corpus_all_stops_stem, 
        control = list(tokenize = trigram_tokenizer))
    df_all_freq_stops_stem_tri <- get_freq_df(tdm_all_tri_stops_stem)
    # - cache 3-gram w/ stops + stem
    saveRDS(df_all_freq_stops_stem_tri, path_df_all_freq_stops_stem_tri)
    rm(tdm_all_tri_stops_stem); gc()
    # - load trigram rds
    #df_all_freq_stops_tri <- readRDS(path_df_all_freq_stops_tri)
    #df_all_freq_stops_stem_tri <- readRDS(path_df_all_freq_stops_stem_tri)
    
    # - four-gram tokenization, w/ stop words
    tdm_all_stops_four <- tm::TermDocumentMatrix(corpus_all_stops, 
        control = list(tokenize = fourgram_tokenizer))
    df_all_freq_stops_four <- get_freq_df(tdm_all_stops_four)
    # - cache 4gram w/ stop words
    saveRDS(df_all_freq_stops_four, path_df_all_freq_stops_four)
    rm(tdm_all_stops_four); gc()
    # - 4gram tokenization: w/ stops + stemming
    tdm_all_stops_stem_four <- tm::TermDocumentMatrix(corpus_all_stops_stem, 
        control = list(tokenize = fourgram_tokenizer))
    df_all_freq_stops_stem_four <- get_freq_df(tdm_all_stops_stem_four)
    # - cache 4gram w/ stop + stem
    saveRDS(df_all_freq_stops_stem_four, path_df_all_freq_stops_stem_four)
    rm(tdm_all_stops_stem_four); gc()
    # - load rds
    #df_all_freq_stops_four <- readRDS(path_df_all_freq_stops_four)
    #df_all_freq_stops_stem_four <- readRDS(path_df_all_freq_stops_stem_four)
    
    # - five-gram tokenization
    tdm_all_stops_stem_five <- tm::TermDocumentMatrix(corpus_all_stops_stem, 
        control = list(tokenize = fivegram_tokenizer))
    df_all_freq_stops_stem_five <- get_freq_df(tdm_all_stops_stem_five)
    # - cache 
    saveRDS(df_all_freq_stops_stem_five, path_df_all_freq_stops_stem_five)
    rm(tdm_all_stops_stem_five); gc()
    
    # - testing: check that bigram and unigram produce the same w1 count
    df_all_freq_uni %>% dplyr::filter(ngram == "case")
    df_all_freq_uni %>% dplyr::filter(grepl("^case$", ngram))
    colSums(df_all_freq_bi %>% dplyr::filter(grepl("^case ", ngram)) %>% dplyr::select(freq))
    
    # - vector is a restricted structure where all components have to be of the
    #   same type, but a list is unrestricted. Hence, list of list rather than
    #   vector of lists.
    quiz_input <- list(
        # - bigram w/ stop words removed
        q1 = list(
            sent="The guy in front of me just bought a pound of bacon, a bouquet, and a case of",
            opts=c("beer","soda","pretzels","cheese")),
        # - bigram w/ stop words removed
        q2 = list(
            sent="You're the reason why I smile everyday. Can you follow me please? It would mean the",
            opts=c("world","universe","most","best")),
        # - unigram
        q3 = list(
            sent="Hey sunshine, can you follow me and make me the",
            opts=c("happiest","smelliest","bluest","saddest")),
        # - NOT MODELED!
        q4 = list(
            sent="Very early observations on the Bills game: Offense still struggling but the",
            opts=c("players","defense","crowd","referees")),
        # - NOT MODELED!
        q5 = list(
            sent="Go on a romantic date at the",
            opts=c("movies","grocery","beach","mall")),
        # - trigram
        q6 = list(
            sent="Well I'm pretty sure my granny has some old bagpipes in her garage I'll dust them off and be on my",
            opts=c("phone","horse","motorcycle","way")),
        # - bigram model, w/ stop words removed. seems to give good results
        q7 = list(
            sent="Ohhhhh #PointBreak is on tomorrow. Love that film and haven't seen it in quite some",
            opts=c("thing","time","years","weeks")),
        # - bigram w/ stop words removed
        q8 = list(
            sent="After the ice bucket challenge Louis will push his long wet hair out of his eyes with his little",
            opts=c("eyes","fingers","ears","toes")),
        # - bigram w/ stop words removed (trigram, unigram support too)
        q9 = list(
            sent="Be grateful for the good times and keep the faith during the",
            opts=c("hard","worse","bad","sad")),
        # - NOT MODELED!
        q10 = list(
            sent="If this isn't the cutest thing you've ever seen, then you must be",
            opts=c("insane","asleep","callous","insensitive")) 
    )
    
    # - load data
    pryr::mem_used()
    df_all_freq_uni <- readRDS(path_df_all_freq_uni)
    df_all_freq_bi <- readRDS(path_df_all_freq_bi)
    df_all_freq_stops_bi <- readRDS(path_df_all_freq_stops_bi)
    df_all_freq_stops_tri <- readRDS(path_df_all_freq_stops_tri)
    df_all_freq_stops_stem_tri <- readRDS(path_df_all_freq_stops_stem_tri)
    df_all_freq_stops_four <- readRDS(path_df_all_freq_stops_four)
    df_all_freq_stops_stem_four <- readRDS(path_df_all_freq_stops_stem_four)
    df_all_freq_stops_stem_five <- readRDS(path_df_all_freq_stops_stem_five)
    pryr::mem_used()
    rm(df_all_freq_uni, df_all_freq_bi, df_all_freq_stops_tri, df_all_freq_stops_stem_tri,df_all_freq_stops_four,df_all_freq_stops_stem_five); gc()
    
    
    q <- "q1"
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_stem_five, n=5, rm_stop_words=FALSE, stem=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_four, n=4, rm_stop_words=FALSE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_stem_four, n=4, rm_stop_words=FALSE, stem=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_tri, n=3, rm_stop_words=FALSE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_stem_tri, n=3, rm_stop_words=FALSE, stem=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_bi, n=2, rm_stop_words=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_uni, n=1, rm_stop_words=TRUE)
    
    
    
    # TO DO
    # - maintain end of sentences so ngrams are just created inter-sentence, not across
    # - trigrams without stop words removed
    # - think about some backoff technique since bigrams (with stop words rm)
    #   seemed to work ok for about half the examples
    # - pruning
    #   > only store ngrams with count > x, start with x=1 (singleton)
    
    
}
# - quiz: week 4
{
    # - Objectives
    #   > preformance
    #       > need to regenerate all the dfs since they have stringsAsFactors = TRUE - DONE!
    #       > sparse terms?
    #       > shouldnt grep, use n-1 gram for denominator. hence eveything is just O(1) lookups
    #         this certainly required a bit more memory tho
    #           > does this work if we remove sparse terms tho?
    #   > compute on entire corpus!
    #       > sentence tokenization
    #   > decide on stemming and stop words
    #       > lets start with the following
    #           > unigram stop words removed, no stemming
    #           > bigram stop words removed, no stemming
    #           > trigram no stop words removed, no stemming
    #           > 4-gram no stop words removed, no stemming
    #   > deal with unknown words
    #       > there are alot of words with freq 1, can we really just make them all unk?
    #   > smoothing
    #   > implement some sort of backoff method
    #       > 
    #   > interpolation
    #       > 

    
    # - file paths
    path_df_all_freq_uni <- file.path(data_cache_dir, "1gram_freq_noStops_noStem.rds")
    path_df_all_freq_bi <- file.path(data_cache_dir, "2gram_freq_noStops_noStem.rds")
    
    path_df_all_freq_stops_tri <- file.path(data_cache_dir, "3gram_freq_stops_noStem.rds")
    path_df_all_freq_stops_stem_tri <- file.path(data_cache_dir, "3gram_freq_stops_stem.rds")
    path_df_all_freq_stops_four <- file.path(data_cache_dir, "4gram_freq_stops_noStem.rds")
    path_df_all_freq_stops_stem_four <- file.path(data_cache_dir, "4gram_freq_stops_stem.rds")
    path_df_all_freq_stops_stem_five <- file.path(data_cache_dir, "5gram_freq_stops_stem.rds")
    
    # - load data sets
    df_all_freq_uni <- readRDS(path_df_all_freq_uni)
    df_all_freq_bi <- readRDS(path_df_all_freq_bi)
    df_all_freq_stops_bi <- readRDS(path_df_all_freq_stops_bi)
    df_all_freq_stops_tri <- readRDS(path_df_all_freq_stops_tri)
    df_all_freq_stops_stem_tri <- readRDS(path_df_all_freq_stops_stem_tri)
    df_all_freq_stops_four <- readRDS(path_df_all_freq_stops_four)
    df_all_freq_stops_stem_four <- readRDS(path_df_all_freq_stops_stem_four)
    df_all_freq_stops_stem_five <- readRDS(path_df_all_freq_stops_stem_five)
    
    # - convert factors to strings and add n-1
    temp_clean <- function(df)
    {
        # - commenting out since we still want to remove the factor for unigrams
#         n <- length(unlist(strsplit(as.character(df$ngram[1]), " ")))
#         if (n==1) return df
        a <- df %>%
            dplyr::mutate(
                # - remove factor
                ngram = as.character(ngram),
                # - n-1 gram (should be no match for unigrams b/c of the space)
                n_1gram = gsub(" \\w+$", "", ngram)
            )
        return(a)
    }
    # - temp cleaning and save down
    df_all_freq_uni <- temp_clean(df_all_freq_uni)
    df_all_freq_bi <- temp_clean(df_all_freq_bi)
    df_all_freq_stops_tri <- temp_clean(df_all_freq_stops_tri)
    df_all_freq_stops_stem_tri <- temp_clean(df_all_freq_stops_stem_tri)
    df_all_freq_stops_four <- temp_clean(df_all_freq_stops_four)
    df_all_freq_stops_stem_four <- temp_clean(df_all_freq_stops_stem_four)
    df_all_freq_stops_stem_five <- temp_clean(df_all_freq_stops_stem_five)

    # - save down in replacement - * NOTE STILL OTHERS THAT NEED TO BE CLEANED UP *
    saveRDS(df_all_freq_uni, path_df_all_freq_uni)
    saveRDS(df_all_freq_bi, path_df_all_freq_bi)
    saveRDS(df_all_freq_stops_tri, path_df_all_freq_stops_tri)
    saveRDS(df_all_freq_stops_stem_tri, path_df_all_freq_stops_stem_tri)
    saveRDS(df_all_freq_stops_four, path_df_all_freq_stops_four)
    saveRDS(df_all_freq_stops_stem_four, path_df_all_freq_stops_stem_four)
    saveRDS(df_all_freq_stops_stem_five, path_df_all_freq_stops_stem_five)
    
    
    quiz_input <- list(
        # - 
        q1 = list(
            sent="When you breathe, I want to be the air for you. I'll be there for you, I'd live and I'd",
            opts=c("die","eat","give","sleep")),
        # - 
        q2 = list(
            sent="Guy at my table's wife got up to go to the bathroom and I asked about dessert and he started telling me about his",
            opts=c("spiritual","marital","horticultural","financial")),
        # - 
        q3 = list(
            sent="I'd give anything to see arctic monkeys this",
            opts=c("decade","weekend","month","morning")),
        # - 
        q4 = list(
            sent="Talking to your mom has the same effect as a hug and helps reduce your",
            opts=c("stress","hapiness","sleepiness","hunger")),
        # - 
        q5 = list(
            sent="When you were in Holland you were like 1 inch away from me but you hadn't time to take a",
            opts=c("minute","walk","look","picture")),
        # - 
        q6 = list(
            sent="I'd just like all of these questions answered, a presentation of evidence, and a jury to settle the",
            opts=c("matter","account","incident","case")),
        # - 
        q7 = list(
            sent="I can't deal with unsymetrical things. I can't even hold an uneven number of bags of groceries in each",
            opts=c("arm","hand","finger","toe")),
        # - 
        q8 = list(
            sent="Every inch of you is perfect from the bottom to the",
            opts=c("side","top","center","middle")),
        # - bigram w/ stop words removed (trigram, unigram support too)
        q9 = list(
            sent="I’m thankful my childhood was filled with imagination and bruises from playing",
            opts=c("outside","weekly","inside","daily")),
        # - 
        q10 = list(
            sent="I like how the same people are in almost all of Adam Sandler's",
            opts=c("stories","novels","pictures","movies"))
    )
    
    # - 
    
    
    q <- "q1"
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_stem_five, n=5, rm_stop_words=FALSE, stem=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_four, n=4, rm_stop_words=FALSE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_stem_four, n=4, rm_stop_words=FALSE, stem=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_tri, n=3, rm_stop_words=FALSE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_stops_stem_tri, n=3, rm_stop_words=FALSE, stem=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_bi, n=2, rm_stop_words=FALSE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_bi, n=2, rm_stop_words=TRUE)
    ngram_model(sentence=quiz_input[[q]]$sent, next_words=quiz_input[[q]]$opts, 
        df_freq=df_all_freq_uni, n=1, rm_stop_words=TRUE)
    
    
    
    
}




# - task 1: playground
{    
    # - combine into a single charecter vector
    tweets <- paste(dat_us_twit, collapse = " ")
    # - convert from charecter class to string class inorder to use NLP library
    tweets <- as.String(tweets)
    # - word and sentence annotators
    #   mark the places in the string where words and sentences start and end
    word_ann <- Maxent_Word_Token_Annotator()
    sent_ann <- Maxent_Sent_Token_Annotator()
    # - First we have to determine where the sentences are, then we can 
    #   determine where the words are. The result is a annotation object
    tweets_annotations <- annotate(tweets, list(sent_ann, word_ann))
    # - create what the NLP package calls an AnnotatedPlainTextDocument
    tweets_doc <- AnnotatedPlainTextDocument(tweets, tweets_annotations)
    # - 
    sents(tweets_doc) %>% head(2)
    # - 
    words(tweets_doc) %>% head(9)
    # - entity annotator - named entity recognition (NER)
    person_ann <- Maxent_Entity_Annotator(kind = "person")
    location_ann <- Maxent_Entity_Annotator(kind = "location")
    organization_ann <- Maxent_Entity_Annotator(kind = "organization")
    # - 
    pipeline <- list(sent_ann,
                 word_ann,
                 person_ann,
                 location_ann,
                 organization_ann)
    tweets_annotations <- annotate(tweets, pipeline)
    tweets_doc <- AnnotatedPlainTextDocument(tweets, tweets_annotations)
    
    # Extract entities from an AnnotatedPlainTextDocument
    entities <- function(doc, kind) {
      s <- doc$content
      a <- annotations(doc)[[1]]
      if(hasArg(kind)) {
        k <- sapply(a$features, `[[`, "kind")
        s[a[k == kind]]
      } else {
        s[a[a$type == "entity"]]
      }
    }
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # - load the data into a vector (in memory)
    dat_us_twit <- readLines(file_us_twit, skipNul = F, n = 5)
    # - instantiate tm corpus
    vcorp_tweet <- tm::VCorpus(tm::VectorSource(dat_us_twit), readerControl=list(language="en"))
    # - concise overview of corpus
    print(vcorp_tweet)
    # - detailed view of corpus
    tm::inspect(vcorp_tweet)
    # - print out a few documents
    writeLines(as.character(vcorp_tweet[[2]]))
    lapply(vcorp_tweet, as.character)
    # - transformation: remove extra white space
    vcorp_tweet <- tm::tm_map(vcorp_tweet, stripWhitespace)
    # - transformation: convert to lower case
    vcorp_tweet <- tm::tm_map(vcorp_tweet, tm::content_transformer(tolower))
    # - transformation: remove stopwords
    #   this adds extra whitespaces, so run that removal after this
    #vcorp_tweet <- tm::tm_map(vcorp_tweet, tm::removeWords, tm::stopwords(kind = "en"))
    # - stemming
    lapply(tm_map(vcorp_tweet, stemDocument), as.character)
    # - filtering (based on meta data)
    meta(vcorp_tweet, "id") == "3"
    # - term document matrix (stored as sparse matrices)
    #   TermDocumentMatrix and DocumentTermMatrix (depending on whether you want 
    #   terms as rows and documents as columns, or vice versa) 
    dtm <- tm::DocumentTermMatrix(vcorp_tweet)
    tm::inspect(dtm)
    # - find words that appear atleast 2 times in out doc matrix
    tm::findFreqTerms(dtm, 2)
    # - find associations (terms that correlate)
    tm::findAssocs(dtm, "smile", 0.8)
    # - remove sparse terms (can dramatically reduce matrix size)
    inspect(removeSparseTerms(dtm, 0.4))
    # - dictionary (only terms in the dict appear in the matrix)
    #   hence we can focus on specific terms for distinct text mining context
    inspect(tm::DocumentTermMatrix(vcorp_tweet, list(dictionary=c("for","more","you"))))
    
    
    # - document term matrix (x=doc, y=word)
    dtm <- tm::DocumentTermMatrix(dat_us_tweet)
    # - 
    a <- colSums(as.matrix(dtm))
    a_ord <- order(a)
    a[tail(a_ord,20)]
    # - words with at least x occurances
    tm::findFreqTerms(dtm, lowfreq=50)
    # - distribution of term frequencies
    table(a)
    hist(table(a), breaks=50)
    # - word associations
    #   if two words always appear together than the correlation would be 1.0
    #   and if they never appear together the correlation would be 0
    tm::findAssocs(dtm, "computer", corlimit=0.4)
    # - correlation network map
    plot(dtm, terms=tm::findFreqTerms(dtm, lowfreq=100), corThreshold=0.2)
    # - 
    wordcloud(names(a), a, min.freq=100)
    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # - word tokens
    example_text = "Hello Gus, how are you doing? Why are you using R? Python might be better."
    example_text_string <- NLP::as.String(example_text)
    # - create annotators
    word_ann <- openNLP::Maxent_Word_Token_Annotator(language = "en", probs = FALSE, model = NULL)
    sent_ann <- openNLP::Maxent_Sent_Token_Annotator(language = "en", probs = FALSE, model = NULL)
    # - 
    ex_txt_annotations <- NLP::annotate(example_text_string, list(sent_ann, word_ann))
    ex_txt_annotations
    # - 
    txt_doc <- NLP::AnnotatedPlainTextDocument(example_text_string, ex_txt_annotations)
    NLP::sents(txt_doc)
    NLP::words(txt_doc)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # - open connection
    con_tweet <- file(file_us_twit, "r") 
    # - 
    
    # - close file connection
    close(con_tweet)
    showConnections(all = FALSE)
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    corpus_all <- tm::Corpus(tm::DirSource(directory = small_data_dir, encoding = "UTF-8"))
    meta(corpus_all, "id")
    doc_id <- "en_US.news.txt"
    row_n <- 10
    corpus_all[[doc_id]][["content"]][row_n]
    Encoding(corpus_all[[doc_id]][["content"]][row_n])
    
    tst_corp <- Corpus(VectorSource((corpus_all[[doc_id]][["content"]][row_n])))
    tst_corp[[1]][["content"]]
    Encoding(tst_corp[[1]][["content"]])
    
    tst_corp_clean2 <- clean_corpus(tst_corp)
    tst_corp_clean2[[1]][["content"]]
    
    onlyAlpha <- content_transformer(function(x) stringi::stri_replace_all_regex(x,"[^\\p{L}\\s[']]+",""))
    tst_corp <- tm_map(tst_corp, onlyAlpha)
    tst_corp[[1]][["content"]]
    
    rm_unicode <- function(x) iconv(x, from="UTF-8", to="ASCII", sub="")
    tst_corp <- tm::tm_map(tst_corp, tm::content_transformer(rm_unicode))
    tst_corp[[1]][["content"]]
    
    
    tst_corp_clean <- tst_corp
    Encoding(tst_corp_clean[[1]][["content"]])
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::content_transformer(tolower))
    tst_corp_clean[[1]][["content"]]
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::removeWords, tm::stopwords("english"))
    tst_corp_clean[[1]][["content"]]
    #tst_corp_clean <- tm::tm_map(tst_corp_clean, trans_gsub, "[[:punct:]]", "")
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::removePunctuation)
    # - encoding is switches from UTF-8 to unknow after punctuation is removed
    #   this also removes at the UTF charecters
    tst_corp_clean[[1]][["content"]]
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::removeWords, dat_profanity)
    tst_corp_clean[[1]][["content"]]
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::removeNumbers)
    tst_corp_clean[[1]][["content"]]
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::stripWhitespace)
    tst_corp_clean <- tm::tm_map(tst_corp_clean, trans_gsub, "^\\s+|\\s+$", "")
    tst_corp_clean[[1]][["content"]]
    
    rm_unicode <- function(x) iconv(x, from="UTF-8", to="ASCII", sub="")
    tst_corp_clean <- tm::tm_map(tst_corp_clean, tm::content_transformer(rm_unicode))
    tst_corp_clean[[1]][["content"]]
    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DISTINCT CHARACTERS
    
    corpus_all <- tm::Corpus(tm::DirSource(directory = small_data_dir))
    # corpus_news <- tm::VCorpus(tm::VectorSource(dat_news))
    # corpus_blog <- tm::VCorpus(tm::VectorSource(dat_blog))
    # corpus_twit <- tm::VCorpus(tm::VectorSource(dat_twit))
    
    corpus_all <- clean_corpus(corpus_all)
    
    tdm_all <- tm::TermDocumentMatrix(corpus_all)
    df <- data.frame(qdap::dist_tab(unlist(strsplit(rownames(tdm_all),"")))) %>%
        dplyr::arrange(desc(freq))
    
    ggplot(data=df, aes(x=reorder(interval,-percent),y=percent)) + geom_point(stat="identity")
    
    iconv(tst_corp_clean2[[1]][["content"]], from="UTF-8", to="ASCII", sub="")
    qdap::dist_tab(unlist(stringr::str_split(tst_corp_clean2[[1]][["content"]],"")))
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 'u u u' in NEWS
    
    corpus_news <- tm::VCorpus(tm::VectorSource(dat_news))
    corpus_news <- clean_corpus(corpus_news)
    
    # - define tokenizers
    bigram_tokenizer <- function(x) unlist(
        lapply(NLP::ngrams(NLP::words(x), 2), paste, collapse = " "), use.names = FALSE)
    trigram_tokenizer <- function(x) unlist(
        lapply(NLP::ngrams(NLP::words(x), 3), paste, collapse = " "), use.names = FALSE)
    
    tdm_news <- tm::TermDocumentMatrix(corpus_news)
    tdm_news_bi <- tm::TermDocumentMatrix(corpus_news, control = list(tokenize = bigram_tokenizer))
    tdm_news_tri <- tm::TermDocumentMatrix(corpus_news, control = list(tokenize = trigram_tokenizer))
    
    freq <- rowSums(as.matrix(tdm_news_tri))
    df <- data.frame(word=names(freq), freq=freq) %>% dplyr::top_n(20, freq) %>% dplyr::arrange(desc(freq))
    df
    
    a <- unlist(lapply(corpus_news, as.character))
    grep("u u", a)
    a[1721]

}

# - rprof testing
{
    # - 
    funAgg = function(x) 
    {
         # initialize res 
         res = NULL
         n = nrow(x)
        
         for (i in 1:n) {
            if (!any(is.na(x[i,]))) res = rbind(res, x[i,])
         }
         res
    }
    
    # - 
    funLoop = function(x) 
    {
         # Initialize res with x
         res = x
         n = nrow(x)
         k = 1
        
         for (i in 1:n) {
            if (!any(is.na(x[i,]))) {
               res[k, ] = x[i,]
               k = k + 1
            }
         }
         res[1:(k-1), ]
    }
    
    # - 
    funApply = function(x) 
    {
        drop = apply(is.na(x), 1, any)
        x[!drop, ]
    }
    
    # - 
    funOmit = function(x) 
    {
        # The or operation is very fast, it is replacing the any function
        # Also note that it doesn't require having another data frame as big as x
        
        drop = F
        n = ncol(x)
        for (i in 1:n)
            drop = drop | is.na(x[, i])
        x[!drop, ]
    }
    
    # - make up large test case
    xx = matrix(rnorm(2000000),100000,20)
    xx[xx>2] = NA
    x = as.data.frame(xx)
    
    # - call the R code profiler and give it an output file to hold results
    Rprof(file.path("profile", "profile_exampleAgg.out"))
    # - call the function to be profiled
    y = funAgg(xx)
    Rprof(NULL)
    
    Rprof(file.path("profile", "profile_exampleLoop.out"))
    y = funLoop(xx)
    Rprof(NULL)
    
    Rprof(file.path("profile", "profile_exampleApply.out"))
    y = funApply(xx)
    Rprof(NULL)
    
    Rprof(file.path("profile", "profile_exampleOmit.out"))
    y = funOmit(xx)
    Rprof(NULL)
    
    # - clean up
    rm(funAgg, funLoop, funApply, funOmit, xx, x)
    
    # summarize the results
    summaryRprof(file.path("profile", "profile_exampleAgg.out"))
}


# - main
{
    # - define paths to data
    root_dir <- "/Users/Gus/Google Drive/dev/coursera/data_sciences/10_capstone"
    setwd(root_dir)
    file_en_us_blog <- file.path(root_dir, "final/en_US", "en_US.blogs.txt")
    file_en_us_twit <- file.path(root_dir, "final/en_US", "en_US.twitter.txt")
    file_en_us_news <- file.path(root_dir, "final/en_US", "en_US.news.txt")
    
    # - load vector of profanity
    #   profanity source: http://fffff.at/googles-official-list-of-bad-words/
    profanity_en_us <- file.path(root_dir, "profanity_list_en.txt")
    dat_en_us_profanity <- readLines(profanity_en_us)
    
    # - pre-process data
    n_lines <- 5
    corp_us_tweet <- get_clean_corpus(file_en_us_twit, dat_en_us_profanity, n_lines=n_lines)
    tdm_us_tweet <- get_tokenization(corp_us_tweet, n_gram=3)
    
    
}


#' construct corpus and clean
#' 
#' @param filepath - (string) filepath to text data
#' @return corpus
get_clean_corpus <- function(filepath, dat_profanity, language="en", n_lines=-1L)
{
    # - Tasks to accomplish:
    #   > Tokenization - identifying appropriate tokens such as words, 
    #       punctuation, and numbers. Writing a function that takes a file as 
    #       input and returns a tokenized version of it.
    #   > Profanity filtering - removing profanity and other words you do not 
    #       want to predict.
    
    # - required libraries
    require(tm)             # - comprehensive text mining framework
    
    # - testing (comment out when function goes live)
#     filepath <- file_en_us_twit
#     language <- "en"
#     n_lines <- 1000
#     dat_profanity <- dat_en_us_profanity
#     dat <- c("?hello, gus!","whats-up-bro :)", "Pimps. pimps. pimps.", "Fuck bitches get money yo!")

    # - check files exists
    if ( ! file.exists(filepath) )
    {
        stop(paste("data file does not exist:",filepath))
    }
    if ( ! is.vector(dat_profanity) )
    {
        stop("profanity data must be a vector")
    }
    
    # - read in data
    dat <- readLines(filepath, skipNul = FALSE, n = n_lines)
    
    # - construct corpus (in memory)
    dat_corpus <- tm::VCorpus(
        tm::VectorSource(dat), 
        readerControl=list(language=language)
    )
    
    # - find non-english charecters
    #   we can just remove the charecters since that would leave us with word
    #   fragments, we should subset the words and add them to stop word list
#     idx_non_ascii <- grep("__NON_ASCII__", iconv(dat, "UTF-8", "ASCII", sub="__NON_ASCII__"))
#     dat[idx_non_ascii]
#     dat[15]
#     iconv(dat[15], "latin1", "ASCII", sub="")
    # - can we tokenize and remove the matches??
    # - do we really need to do this? all these words should have very low
    #   associations and hence not come up in prediction?
    
    # - debug: print corpus
#     lapply(dat_corpus, as.character)
    
    # - delete raw data not that corpus is constructed
    rm(dat); gc()    
    
    # - testing: check supported
    #tm::getTransformations()
    #tm::getTokenizers()
    
    # - define two transforms
    trans_gsub <- tm::content_transformer(
        function(x, pattern, replacement) gsub(pattern, replacement, x))
    
    # - transform: remove puntuations
    #   list: https://stat.ethz.ch/R-manual/R-devel/library/base/html/regex.html
    dat_corpus <- tm::tm_map(dat_corpus, trans_gsub, "[[:punct:]]", " ")
    # - transform: remove punctuation
    #   puntuation gets replaces with empty string, not words
    #   hence 'whats-up-man' becomes 'whatsupman'
    #dat_corpus <- tm::tm_map(dat_corpus, tm::removePunctuation)
    
    # - does it make sense to remove all punctuation? how can you sentence
    #   tokenize then? You will might get alot of word associations between
    #   end of one sentence and start of another.
    
    # - transform: convert to lower case
    dat_corpus <- tm::tm_map(dat_corpus, tm::content_transformer(tolower))
    
    # - transform: remove numbers
    dat_corpus <- tm::tm_map(dat_corpus, tm::removeNumbers)
    
    # - find all non-ascii words
    
    # - transform: remove stop words
    #going to hold off for now
    #dat_corpus <- tm::tm_map(dat_corpus, tm::removeWords, tm::stopwords(kind=language))
    
    # - transform: remove profanity (perform after lowercase)
    dat_corpus <- tm::tm_map(dat_corpus, tm::removeWords, dat_profanity)
    
    # - stemming
    #going to hold off for now
    
    # - transformation: remove extra white spaces and trim (final step)
    #   tm transform doesnt remove training or leading whitespace
    dat_corpus <- tm::tm_map(dat_corpus, tm::stripWhitespace)
    dat_corpus <- tm::tm_map(dat_corpus, trans_gsub, "^\\s+|\\s+$", "")
    
    # - remove non-ascii words

    
    
    # - return cleaned corpus
    invisible(dat_corpus)
}


#' tokenize a corpus
#' 
#' @param corpus - (Corpus) clean corpus
#' @param n_gram - (integer) clean corpus
#' @return tdm
get_tokenization <- function(corpus, n_gram=1)
{
    # - required libraries
    require(tm)             # - comprehensive text mining framework
    require(tau)            # - text analysis utilities
    
    # - testing
#     corpus <- corp_us_tweet
#     n_gram <- 3
    
    # - define n-gram tokenization function
    n_gram_tokenizer <- function(x) 
        unlist(
            lapply(
                NLP::ngrams(NLP::words(x), n_gram), 
                paste, 
                collapse = " "
            ), 
            use.names = FALSE
        )
    
    tdm <- tm::TermDocumentMatrix(corpus, control = list(tokenize = n_gram_tokenizer))
    
    # - return
    invisible(tdm)
}


#' predict the next word in the sentence using an n-gram model 
#' (maximum likelihood estimation).
#' 
#' Bigram Model: P(w2|w1) =  count(w1,w2)/count(w1)
#'   > count(w1): should be able to get this from both a unigram and bigram
#'     in bigram, we just need to grep for "^w1 "
#' Trigram Model: P(w3|w1,w2) = count(w1,w2,w3) / count(w1,w2)
#' 
#' @param sentence - (string) input we are trying to predict the next word for
#' @param next_words - (vector strings) options for next words
#' @param df_freq - (data.frame) bigram frequency table
#' @param n - (numeric) n-gram
#' @return (void) prints results to screen
ngram_model <- function(sentence, next_words, df_freq, n=2, rm_stop_words=TRUE,
    stem=FALSE)
{
    start_time <- Sys.time()
    
    # - load required libraries
    require(tm)
    require(dplyr)
    require(stringr)
    
    # - testing
#     sentence <- "When you breathe, I want to be the air for you. I'll be there for you, I'd live and I'd"
#     next_words <- c("die","eat","give","sleep")
#     df_freq <- df_all_freq_uni
#     n <- 1
#     rm_stop_words <- TRUE
#     stem <- FALSE
    
    # - convert the sentence to a corpus
    corpus_sent <- tm::VCorpus(tm::VectorSource(c(sentence)))
    # - clean the corpus
    corpus_sent <- clean_corpus(corpus_sent, rm_stop_words, stem)
    #lapply(corpus_sent, as.character)
    sent_vector <- unlist(lapply(corpus_sent, as.character))
    sent_tail <- paste(tail(stringr::str_split(sent_vector, pattern=" ")[[1]],n-1), collapse=" ")
    cat(paste0(n,"-gram (stem=",stem,") :"),sent_vector,"______","\n")
    
    # - setup regex
    #w1 <- paste0("^",sent_tail, " ")
    
    # - compute all n-grams that start with word (denominator)
    #df_freq_match <- df_freq %>% dplyr::filter(grepl(w1, ngram))
    df_freq_match <- df_freq %>% dplyr::filter(n_1gram==sent_tail)
    denom <- as.numeric(colSums(df_freq_match %>% dplyr::select(freq)))
    
    # - special handling for unigrams
    if (n==1)
    {
        #w1 <- ""
        denom <- nrow(df_freq)
    }
    
    # - stem the input words
    if ( stem )
    {
        corpus_next <- tm::VCorpus(tm::VectorSource(c(next_words)))
        corpus_next <- tm::tm_map(corpus_next, tm::stemDocument)
        next_words <- unlist(lapply(corpus_next, as.character))
    }
    
    # - loop over next words and check frequency (print to screen)
    for (opt_cv in next_words)
    {
        # - not only is this grep slow but it is wrong since it doenst look for end of sent too
#         num <- as.numeric(colSums(df_freq %>% 
#             dplyr::filter(grepl(paste0(w1,opt_cv), ngram)) %>% 
#             dplyr::select(freq)))
        # - testing
#         df_temp <- df_freq %>% dplyr::filter(grepl(paste0(w1,opt_cv), ngram))
#         print(head(df_temp))
#         num <- as.numeric(colSums(df_temp %>% dplyr::select(freq)))
        
        # - sequence to lookup in df
        num <- NULL
        if ( n==1 )
        {
            num <- as.numeric(colSums(df_freq %>% 
                dplyr::filter(ngram == opt_cv) %>%
                dplyr::select(freq)))
        }
        else
        {
            num <- as.numeric(colSums(df_freq_match %>% 
                dplyr::filter(ngram == paste(sent_tail,opt_cv)) %>%
                dplyr::select(freq)))
        }

        # - compute percentages
        fraction <- num / denom
        cat("  > ",opt_cv,": ",num," / ",denom," => ",100*fraction, ", log-prob: ",
            log2(fraction) , "%\n", sep="")
    }
    
    # - print the top probabilities regardless of inputs
    cat("  > top n-grams:\n")
    if (n==1 || nrow(df_freq_match)==0)
    {
        head(df_freq)
    }
    else
    {
        head(df_freq_match)
    }
    
    end_time <- Sys.time()
    cat("function run time: ", end_time-start_time," seconds \n")
    
    # - void return
    #return(NULL)
}


#' load python processed ngram data
#' 
#' @param filepath - (string) path to csv file
#' @return (data.frame) {ngram, freq, n, n_cum_pct, freq_pct, freq_cum_pct, n_1gram}
load_py_proc_data <- function(filepath)
{
    require(readr)
    require(dplyr)
    
    # - testing
    filepath <- file.path(data_cache_dir, "3grams.csv")
    
    
    # need to skip nulls for the moment (few in twitter than need to be stripped out in py)
    df_py_raw <- read.csv(filepath, stringsAsFactors=F, 
        colClasses=c("character","integer"), skipNul = T) %>%
        dplyr::arrange(desc(freq)) %>%
        dplyr::mutate(
            n = 1:n(),
            n_cum_pct = n / n(),
            ttl_freq = sum(freq),
            freq_pct = ifelse(freq==0,0,freq/ttl_freq),
            freq_cum_pct = cumsum(freq_pct),
            n_1gram = gsub(" \\w+$", "", ngram)
        ) %>%
        dplyr::select(-ttl_freq)
    
}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# - TESTING
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

c <- c("I am Sam", "Sam I am", "I do not like green eggs and ham")
corp <- tm::VCorpus(tm::VectorSource(c))
tdm_bigram <- tm::TermDocumentMatrix(corp, control = list(tokenize = bigram_tokenizer))
df_bigram <- get_freq_df(tdm_bigram)
str(df_bigram)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

corp_twit <- tm::Corpus(tm::DirSource(sample_dir, encoding="UTF-8", pattern="news"))

corpus_twit_slim <- tm::VCorpus(tm::VectorSource(corp_twit[[1]][["content"]][c(2)]))
lapply(corpus_twit_slim, as.character)

corpus_twit_slim_clean <- clean_corpus(corpus_twit_slim, rm_stop_words=F, stem=F)
lapply(corpus_twit_slim_clean, as.character)

tdm_bigram <- tm::TermDocumentMatrix(corpus_twit_slim_clean, control = list(tokenize = bigram_tokenizer))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

length(tdm_all_tri[["dimnames"]][["Terms"]])
grep("case of ", tdm_all_tri[["dimnames"]][["Terms"]], value=T)
df_all_freq <- get_freq_df(tdm_all_tri)
head(df_all_freq[grep("^case of ", df_all_freq$ngram),])


corpus_news <- tm::Corpus(
    tm::DirSource(sample_dir, encoding="UTF-8", pattern="news"), 
    readerControl = list(language="en_US"))

corpus_news_slim <- tm::VCorpus(tm::VectorSource(corpus_news_clean[[1]][["content"]][51002]))


corpus_news_clean <- clean_corpus(corpus_news)
corpus_news_slim <- clean_corpus(corpus_news_slim, rm_stop_words = FALSE)

# - define n-gram tokenizers
bigram_tokenizer <- function(x) unlist(
    lapply(NLP::ngrams(NLP::words(x), 2), paste, collapse = " "), use.names = FALSE)
trigram_tokenizer <- function(x) unlist(
    lapply(NLP::ngrams(NLP::words(x), 3), paste, collapse = " "), use.names = FALSE)
fourgram_tokenizer <- function(x) unlist(
    lapply(NLP::ngrams(NLP::words(x), 4), paste, collapse = " "), use.names = FALSE)

tdm_news_tri <- tm::TermDocumentMatrix(corpus_news_clean, 
    control = list(tokenize = trigram_tokenizer))
length(tdm_news_tri[["dimnames"]][["Terms"]])
grep("case of ", tdm_news_tri[["dimnames"]][["Terms"]], value=T)
df_freq <- get_freq_df(tdm_news_tri)
head(df_freq[grep("^case of ", df_freq$ngram),])


tdm_news_slim_tri <- tm::TermDocumentMatrix(corpus_news_slim, 
    control = list(tokenize = trigram_tokenizer))
tdm_news_slim_tri[["dimnames"]][["Terms"]]
grep("case of", tdm_news_slim_tri[["dimnames"]][["Terms"]], value=T)

df_freq <- get_freq_df(tdm_all)
    






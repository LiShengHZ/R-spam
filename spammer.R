library(tm)
library(ggplot2)

# train classifier

spam.path <- "data/spam/"
spam2.path <- "data/spam_2/"
easyham.path <- "data/easy_ham/"
easyham2.path <- "data/easy_ham_2/"
hardham.path <- "data/hard_ham/"
hardham2.path <- "data/hard_ham_2/"


get.msg <- function(path) {
	con <- file(path, open="rt", encoding="latin1")
	text <- readLines(con)
	# email body beginds after first full line break
	from <- which(text=="")[1]+1
	to <- length(text)
	msg <- text[seq(from, to, 1)]
	close(con)
	return(paste(msg, collapse="\n"))
}

# create one vector w/ all text content
spam.docs <- dir(spam.path)
# ignore some dataset files
spam.docs <- spam.docs[which(spam.docs!="cmds")]

# will create one huge vector and set filenames as names
all.spam <- sapply(spam.docs, function(p) get.msg(paste(spam.path, p, sep="")))
all.spam <- all.spam[seq(1, 500, 1)]

# get Term document matrix (TDM) [n terms; m docs]
get.tdm <- function(doc.vec) {
	doc.corpus <- Corpus(VectorSource(doc.vec))
	control <- list(stopwords=TRUE, removePunctuation=TRUE, removeNumbers=TRUE, minDocFreq=2)
	doc.tdm <- TermDocumentMatrix(doc.corpus, control)
	return(doc.tdm)
}

spam.tdm <- get.tdm(all.spam)


# now begin build classifier
# 1. create training data from spam
# construct data frame that contains all observed probabilities for
# each term (given that we now its spam)
spam.matrix <- as.matrix(spam.tdm)
spam.counts <- rowSums(spam.matrix)
spam.df <- data.frame(cbind(names(spam.counts), as.numeric(spam.counts)), stringsAsFactors = FALSE)
names(spam.df) <- c("term", "frequency")
spam.df$frequency <- as.numeric(spam.df$frequency)

# what is the percentage of documents that this term does appear
# if I take any spam term, in how many percent of the documents does 
# this term appear in.
# in how many docs does this term appear (percent of docs)
spam.occurrence <- sapply(1:nrow(spam.matrix), function(i) {
	length(which(spam.matrix[i,] > 0)) / ncol(spam.matrix)
})

# if I take any spam term, how large is the percentage of it being
# the current term
# how often does this term appear (percent of all the words)
spam.density <- spam.df$frequency/sum(spam.df$frequency)

# add new vectors to data frame
spam.df <- transform(spam.df, density=spam.density, occurrence=spam.occurrence)

# second approach is better, because some chars like tr appear
# often (html tags), they would destroy the filter weighting, so
# therefore we use occurrence instead of density


# now balance classifier with ham messages
# 2. create training data from ham
easyham.docs <- dir(easyham.path)
easyham.docs <- easyham.docs[which(easyham.docs!="cmds")]
all.easyham <- sapply(easyham.docs, function(p) get.msg(paste(easyham.path, p, sep="")))
all.easyham <- all.easyham[seq(1, 500, 1)]
easyham.tdm <- get.tdm(all.easyham)

easyham.matrix <- as.matrix(easyham.tdm)
easyham.counts <- rowSums(easyham.matrix)
easyham.df <- data.frame(cbind(names(easyham.counts), as.numeric(easyham.counts)), stringsAsFactors = FALSE)
names(easyham.df) <- c("term", "frequency")

easyham.df$frequency <- as.numeric(easyham.df$frequency)

easyham.occurrence <- sapply(1:nrow(easyham.matrix), function(i) {
	length(which(easyham.matrix[i,] > 0)) / ncol(easyham.matrix)
})
easyham.density <- easyham.df$frequency/sum(easyham.df$frequency)
easyham.df <- transform(easyham.df, density=easyham.density, occurrence=easyham.occurrence)


# print sorted by strongest indicators
# print(head(spam.df[with(spam.df, order(-occurrence)),], 20))
# print(head(easyham.df[with(easyham.df, order(-occurrence)),], 20))

# maybe there's something wrong with the data so far


classify.email <- function(path, training.df, prior=0.5, c=1e-6) {
	msg <- get.msg(path)
	msg.tdm <- get.tdm(msg)
	msg.freq <- rowSums(as.matrix(msg.tdm))
	# find intersections of words
	msg.match <- intersect(names(msg.freq), training.df$term)
	if(length(msg.match) < 1) {
		return(prior*c^(length(msg.freq)))
	} else {
		match.probs <- training.df$occurrence[match(msg.match, training.df$term)]
		return (prior * prod(match.probs) * c^(length(msg.freq)-length(msg.match)))
	}
}


# check classifier
hardham.docs <- dir(hardham.path)
hardham.docs <- hardham.docs[which(hardham.docs != "cmd")]

hardham.spamtest <- sapply(hardham.docs, function(p) classify.email(paste(hardham.path, p, sep=""), training.df=spam.df))
hardham.hamtest <- sapply(hardham.docs, function (p) classify.email(paste(hardham.path, p ,sep=""), training.df=easyham.df))

hardham.res <- ifelse(hardham.spamtest > hardham.hamtest, TRUE, FALSE)
print(summary(hardham.res))



# now test the classifier against all messages
spam.classifier <- function (path) {
	pr.spam <- classify.email(path, spam.df)
	pr.ham <- classify.email(path, easyham.df)
	return (c(pr.spam, pr.ham, ifelse(pr.spam > pr.ham, 1, 0)))
}

#print("a")
#ra <- spam.classifier(easyham2.path)
#print("b")
#rb <- spam.classifier(hardham2.path)
#print("c")
#rc <- spam.classifier(spam2.path)

# print(summary(ra))
# print(summary(rb))
# print(summary(rc))
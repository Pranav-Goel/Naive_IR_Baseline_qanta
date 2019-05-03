# Naive_IR_Baseline for [QANTA](https://sites.google.com/view/qanta/home)

This is an **example system *using the evidence files** containing supporting sentences from Wikipedia for every sentence in Quizbowl question to provide additional context. This Naive IR system basically uses the Wiki pages from which the top 5 tfidf sentences for each quizbowl question sentence is coming from.

The Baseline system from which this submission is adapted can be found [here](https://github.com/Pinafore/qanta-codalab). Note that the evaluate.py file has been updated on there (and here) to support the Wiki evidence files, and downloading is now supported in docker-compose (as instructed in the README of the baseline system repo).

This system has been submitted and can be found on the [leaderboard](https://pinafore.github.io/qanta-leaderboard/).

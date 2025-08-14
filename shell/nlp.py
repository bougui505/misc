#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Aug 14 10:05:00 2025

import spacy
import typer

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.callback()
def callback(debug:bool=False):
    """
    Natural Language Processing using spaCy (https://spacy.io) on the command line
    """
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = debug

NLP = spacy.load("en_core_web_sm")

def has_verb(sentence):
    """
    Checks if a spaCy sentence object contains at least one verb.
    """
    # Process the sentence string to get a Doc object
    doc = NLP(sentence)
    
    # Iterate through the tokens in the sentence
    for token in doc:
        # Check if the token's part-of-speech is a verb or an auxiliary verb
        # print(token.text, token.pos_)
        if token.pos_ in ["VERB", "AUX"]:
            return True
    return False

@app.command()
def sbd():
    """
    SBD: sentence boundary detection. This command split the text from stdin to one sentence per line.
    """
    # read the text from stdin
    text = sys.stdin.read()
    if not text:
        typer.echo("No text provided. Please provide text to process.")
        raise typer.Exit(code=1)
    doc = NLP(text)
    for sentence in doc.sents:
        # print the sentence text
        if not has_verb(sentence.text):
            print("")
        print(sentence.text.strip())

if __name__ == "__main__":
    import sys
    app()

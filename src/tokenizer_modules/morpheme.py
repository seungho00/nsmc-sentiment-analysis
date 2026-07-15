from mecab import MeCab

mecab = MeCab()

def morpheme_tokenizer(df, column):
    return df[column].apply(mecab.morphs)
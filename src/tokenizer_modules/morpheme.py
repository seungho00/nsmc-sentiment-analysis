from mecab import MeCab

mecab = MeCab()

def morpheme_tokenizer(df, column="document"):
    return df[column].apply(mecab.morphs)
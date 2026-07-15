def character_level_tokenizer(df, column):
    return df[column].apply(list)
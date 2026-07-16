def character_level_tokenizer(df, column="document"):
    return df[column].apply(list)
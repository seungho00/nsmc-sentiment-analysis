def basic_tokenizer(df, column="document"):
    df = df[column].str.replace(
        r"([.,!?^~;])", r" \1 ", regex=True
    )

    df = df.str.split()
    return df
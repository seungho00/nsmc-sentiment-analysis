def basic_tokenizer(df, column):
    df = df[column].str.replace(
        r"([.,!?^~;])", r" \1 ", regex=True
    )

    df = df.str.split()
    return df
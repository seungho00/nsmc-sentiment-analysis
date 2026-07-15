def sentiment_tokenizer(df, column):
    
    # 감정 표현 정규화
    normalize_patterns = {
        r"\.{3,}": " .. ",
        r"\^{2,}": " ^^ ",
        r";{2,}": " ;; ",
        r",{2,}": " ,, ",
        r"ㅋ{2,}": " ㅋㅋ ",
        r"ㅎ{2,}": " ㅎㅎ ",
        r"ㄷ{2,}": " ㄷㄷ ",
        r"ㅠ{2,}": " ㅠㅠ ",
        r"ㅜ{2,}": " ㅜㅜ ",
        r"ㅡ{2,}": " ㅡㅡ ",
    }

    # 감정 표현은 보호하고 나머지 특수문자만 띄우기
    split_patterns = {
        r"(?<!\.)\.(?!\.)": " . ",
        r"(?<!\^)\^(?!\^)": " ^ ",
        r"(?<!;);(?!;)": " ; ",
        r"(?<!,),(?!,)": " , ",
        r"([!?~])": r" \1 ",
    }

    for pattern, replacement in normalize_patterns.items():
        df[column] = df[column].str.replace(
            pattern, replacement, regex=True
        )

    for pattern, replacement in split_patterns.items():
        df[column] = df[column].str.replace(
            pattern, replacement, regex=True
        )

    return df[column].str.split()
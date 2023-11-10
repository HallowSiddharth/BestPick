import emoji
def give_emoji_free_text(text):
    allchars = [st for st in text]
    emoji_list = [c for c in allchars if c in emoji.EMOJI_DATA]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


print(give_emoji_free_text("😊 hello"))
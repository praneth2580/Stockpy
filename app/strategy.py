def suggest_strategy(prediction):
    if prediction > 19500:
        return "Sell", 500, 200
    elif prediction < 19200:
        return "Buy", 600, 150
    else:
        return "Hold", 100, 50
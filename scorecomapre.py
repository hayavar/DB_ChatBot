def get_score_keys(solution_recommendation):
    best_keys = []
    highest_scores = [-1, -1]  # Initialize with values lower than any possible score

    for key, scores in solution_recommendation.items():
        score1, score2 = scores
        if score1 > highest_scores[0] and score2 > highest_scores[1]:
            highest_scores = [score1, score2]
            best_keys = [key]  # Start a new list of best keys
        elif score1 == highest_scores[0] and score2 == highest_scores[1]:
            best_keys.append(key)  # Add to the list of best keys

    print(f"The best keys with the highest scores ({highest_scores[0]}, {highest_scores[1]}) are: {best_keys}")

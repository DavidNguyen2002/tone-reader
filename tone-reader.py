import praw
import numpy as np
import pandas as pd

# Takes in filename containing Reddit api key and returns Reddit instance
def reddit_api_login(client_key_filename: str) -> praw.Reddit:
    f = open(client_key_filename, "r")
    client_id, client_secret, user_agent = f.readline().split(',')
    f.close()
    
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    return reddit

# Aggregates all comments with '/s' in a Reddit thread to a list of dictionaries
def aggregate_reddit_thread(submission_string: str, reddit: praw.Reddit) -> list[dict]:
    submission = reddit.submission(id=submission_string)

    # List of dictionaries that will be converted to DataFrame
    list_of_dicts = list()

    submission.comments.replace_more(limit=None)
    for comment in submission.comments.list():
        label_idx = comment.body.find('/s')
        
        if label_idx != -1 and comment.body.find('http') == -1:
            comment_clean = clean_comment(comment.body[:label_idx])
            list_of_dicts.append({"Comment": comment_clean, "Label": 1, "Submission String": submission_string})
    
    return list_of_dicts
    
# Take list of dictionaries -> Turn into DataFrame -> Write to csv
def write_to_dataset(list_of_dicts: list[dict]) -> None:
    df = pd.DataFrame(list_of_dicts)
    df.to_csv("labelled_dataset.csv", mode="a", index=False, header=False)

    
# Cleans comment for parsing
# TODO
def clean_comment(comment: str) -> str:
    return comment
    
reddit = reddit_api_login("api-key-reddit")

lst = aggregate_reddit_thread("3g1jfi", reddit)

write_to_dataset(lst)

print(pd.read_csv('labelled_dataset.csv'))
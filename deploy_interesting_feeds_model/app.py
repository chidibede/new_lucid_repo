''' Deploy Interesting feeds system as a web app
    and API
'''
from flask import Flask, render_template, request, jsonify
import pandas as pd
from pandas.io.json import json_normalize
import joblib
import numpy as np

app = Flask(__name__)

def load_datasets(data):
    '''Function to load the dataset
    to be used
    '''
    data = pd.read_csv(data)
    return data

def select_columns(data, columns):
    '''Function to select columns needed
    '''
    data = data.loc[:, columns]
    return data

# Load the data
POSTS = load_datasets('posts.csv')

# Drop duplicates
POSTS = POSTS.drop_duplicates()

def clean_post_data():
    '''
     Function that cleans the post data using regex
    '''
    # Remove html tags
    POSTS['content'] = POSTS['content'].str.replace(r'<[^>]*>', '')

    # Remove white spaces including new lines
    POSTS['content'] = POSTS['content'].str.replace(r'\s', ' ')

    # Remove square brackets
    POSTS['content'] = POSTS['content'].str.replace(r'\[.*?\]', '')

    # Remove image files
    POSTS['content'] = POSTS['content'].str.replace(r'\(.*?\)', '')

# Load the exported system
TOP_FEEDS = joblib.load('model.sav')

# clean post data
clean_post_data()

# Select needed columns
POSTS = select_columns(POSTS, ['id', 'title', 'content'])

# Rename columns
POSTS.columns = ['post_id', 'title', 'content']

# Merge the top feeds with the posts
MERGED_POSTS = pd.merge(POSTS, TOP_FEEDS, on='post_id').drop_duplicates()

# sort
MERGED_POSTS = MERGED_POSTS.sort_values(by='ratings', ascending=False)

def filter_post_length(length):
    '''
    Function that allows us to choose the length we deem short
    '''
    top_posts = MERGED_POSTS[MERGED_POSTS['content'].str.len() > length].reset_index(drop=True)
    return top_posts


# remove short posts
TOP_POSTS = filter_post_length(150)

# top 5 posts
TOP_5_POSTS = np.array(TOP_POSTS.head().loc[:, ['title', 'content', 'ratings']])

# Top 8 posts
TOP_8_POSTS = np.array(TOP_POSTS.head(8).loc[:, ['title', 'content', 'ratings']])

@app.route('/')
def home():
    '''Function to render the home page'''
    return render_template('index.html')

@app.route('/grading_method')
def grading_method():
    '''Function to render the grading page'''
    return render_template('grading_method.html')

@app.route('/api_documentation')
def api_documentation():
    '''Function to render the docs page'''
    return render_template('api_documentation.html')

@app.route('/top_5_feeds')
def top_5_feeds():
    '''Function to return the top 5 posts'''
    try:
        return render_template('top_5_feeds.html', top_5_posts=TOP_5_POSTS)
    except:
        return render_template('top_5_feeds.html', top_5_posts="No top posts")

@app.route('/top_8_feeds')
def top_8_feeds():
    '''Function to return the top 8 posts'''
    try:
        return render_template('top_8_feeds.html', top_8_posts=TOP_8_POSTS)
    except:
        return render_template('top_8_feeds.html', top_8_posts="No top posts")

@app.route('/top_feeds_api', methods=['POST'])
def top_feeds_api():
    '''Function that handles direct api calls
    from another client to display interesting feeds'''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for feed in json_df.feeds:
            number_of_feed = int(feed)

       # top posts
        top_posts_api = filter_post_length(150)
        select_cols = ['title', 'content', 'ratings']
        top_posts_api = np.array(top_posts_api.head(number_of_feed).loc[:, select_cols])

        interesting_feeds = {
            'interesting_feeds': [[post[0], post[1], post[2]] for post in top_posts_api],
        }
        return jsonify(interesting_feeds)
    except:
        print("http error")

if __name__ == "__main__":
    app.run(debug=True)

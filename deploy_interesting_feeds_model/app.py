from flask import Flask, render_template, request, jsonify
import pandas as pd
from pandas.io.json import json_normalize
import joblib
import numpy as np

app = Flask(__name__)

def load_datasets(data):
    data = pd.read_csv(data)
    return data

def select_columns(data, columns):
    data = data.loc[:, columns]
    return data

# Load the data
posts = load_datasets('posts.csv')

# Drop duplicates
posts = posts.drop_duplicates()

def clean_post_data():
    '''
     Function that cleans the post data using regex
    '''
    # Remove html tags
    posts['content'] = posts['content'].str.replace(r'<[^>]*>', '')

    # Remove white spaces including new lines
    posts['content'] = posts['content'].str.replace(r'\s', ' ')

    # Remove square brackets
    posts['content'] = posts['content'].str.replace(r'\[.*?\]', '')

    # Remove image files
    posts['content'] = posts['content'].str.replace(r'\(.*?\)', '')

def filter_post_length(length):
    '''
    Function that allows us to choose the length we deem short
    '''
    top_posts = merged_posts[merged_posts['content'].str.len() > length].reset_index(drop=True)
    return top_posts

# Load the exported system
top_feeds = joblib.load('model.sav')

# clean post data
clean_post_data()

# Select needed columns
posts = select_columns(posts, ['id', 'title', 'content']) 

# Rename columns
posts.columns = ['post_id', 'title', 'content']

# Merge the top feeds with the posts
merged_posts = pd.merge(posts, top_feeds, on='post_id').drop_duplicates()

# sort
merged_posts = merged_posts.sort_values(by='ratings', ascending=False)

# remove short posts
top_posts = filter_post_length(150)

# top 5 posts
top_5_posts = np.array(top_posts.head().loc[:, ['title','content', 'ratings']])

# Top 8 posts
top_8_posts = np.array(top_posts.head(8).loc[:, ['title','content', 'ratings']])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/grading_method')
def grading_method():
    return render_template('grading_method.html')

@app.route('/api_documentation')
def api_documentation():
    return render_template('api_documentation.html')

@app.route('/top_5_feeds')
def top_5_feeds():
    try:
        return render_template('top_5_feeds.html', top_5_posts=top_5_posts)
    except:
        return render_template('top_5_feeds.html', top_5_posts = "No top posts")

@app.route('/top_8_feeds')
def top_8_feeds():
    try:
        return render_template('top_8_feeds.html', top_8_posts=top_8_posts)
    except:
        return render_template('top_8_feeds.html', top_8_posts = "No top posts")

@app.route('/top_feeds_api', methods=['POST'])
def top_feeds_api():
    '''Function that handles direct api calls
    from another client to display interesting feeds'''
    try:
        json_data = request.get_json(force=True)
        json_df = json_normalize(json_data)
        for feed in json_df.feeds:
            number_of_feed =int(feed)
       
       # top posts
        top_posts_api = filter_post_length(150)
        top_posts_api = np.array(top_posts_api.head(number_of_feed).loc[:, ['title','content', 'ratings']])

        interesting_feeds = {
            'interesting_feeds': [[post[0], post[1], post[2]] for post in top_posts_api],
        }
        return jsonify(interesting_feeds)
    except:
        print("http error")

if __name__ == "__main__":
    app.run(debug=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 23:03:21 2019

@author: lumi
"""
#%%
#import the needed modules

import pandas as pd
def popular_posts(notifications):
    #reading the csv file notifications.csv
    notif = pd.read_csv(notifications)
    #counting the times all types of actions is done
    notif['action'].value_counts()
    #dropping irrelevant values
    notif.drop(['id','parent_comment_id', 'comment','type','updated_at', 
                'sender_id', 'status','status_id','action.1','post_id.1',
                'created_at.1','updated_at.1','slug','image','content','tags',
                'title','user_id.1','created_at','id.1'], axis=1, inplace= True)
    #ranking the action types
    action_ranks = {'Commented': 5, 'Love': 4, 'Like': 3, 'Replied': 2}
    #add the ranks to the table
    notif['ratings']= notif['action'].apply(lambda x: action_ranks[x])
    notif.head()
    #group the posts and get the counts with ratings
    counts_in_ratings =  pd.DataFrame(notif.groupby(['post_id'])['ratings'].count())
    # Arrange the output in descending order and viewing head to get the top 10 most popular posts
    return counts_in_ratings.sort_values('ratings', ascending=False)
    
popular_posts('notifications_new.csv')


filename = 'model.sav'
joblib.dump(popular_posts('notifications_new.csv'), filename)
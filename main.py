import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetching and formatting data
data = fetch_movielens(min_rating=4.0)

print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss='warp')
#train model
model.fit(data['train'], epochs=30, num_threads=2)

def sampleRecommendation(model, data, user_ids):
    #number of users and movies in training data
    n_users, n_items = data['train'].shape

    #generate recommendations for each user
    for user_id in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        #movies our model predict they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order from most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #printing out the results
        print("User %s" % user_id)
        print("     Known positives:")
        for x in known_positives:
            print("         %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("         %s" % x)

sampleRecommendation(model, data, [3, 25, 150])



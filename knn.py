from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy.spatial.distance import euclidean
import heapq as hq
from collections import Counter
from sklearn.metrics import accuracy_score

def getNeighbors(instance, numb_neighbors):
    neighbors = []
    for i in xrange(len(x_train)):
        dist = -1*euclidean(x_train[i], instance)
        if len(neighbors) == numb_neighbors:
            if neighbors[0][0] < dist:
                neighbors = neighbors[1:]
                hq.heappush(neighbors, (dist, i))
        else:
            hq.heappush(neighbors, (dist, i))
    return map(lambda x: x[1], neighbors)

def predict(instance, numb_neighbors):
    neighbors = getNeighbors(instance, numb_neighbors)
    votes = []
    for i in xrange(len(neighbors)):
        votes.append(y_train[neighbors[i]])
    return Counter(votes).most_common(1)[0][0]
    
iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

k = 3
y_pred = []
for i in xrange(len(x_test)):
    pred = predict(x_test[i], k)
    y_pred.append(pred)

acc = accuracy_score(y_test, np.asarray(y_pred))
print('Accuracy: %f' % acc)

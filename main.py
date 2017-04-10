import numpy as np
import math
import random
import re
width, height = 15, 15

data_dir = "Data/L30fft_"
data = None

weight_mat = None
alpha = 0.4
radius = max(width, height)

# read in data and conver it to a list of tuples containing
# the list of inputs as the first element and the expected
# output as the second element

def read_data(precision):
    f = open(data_dir + str(precision)+".out")
    f.readline()
    lines = f.readlines()
    #print (re.sub(' +',' ', lines[10]).strip())
    lines = [re.sub('\x1a','  ', x).strip() for x in lines]
    lines = [re.sub(' +',' ', x).strip() for x in lines]
    data_split = np.asarray([x.split() for x in lines if len(x) > 0]).astype(float)
    #print(data_split)
    expected = data_split[:, 0]
    tmp_data = data_split[:, 1:]
    data = list(zip(tmp_data, expected))
    return data

def read_img_data():
    from scipy.misc import imread, imshow
    random_pixels = np.random.rand(50,50,3) 
    return random_pixels

# initializes random weight in a matrix of size m x n
# where m is the number of data points and n is 
# the width * height of the topology
def create_weight_mat(data_dim, net_width, net_height):
    weight_mat = np.random.rand(data_dim, net_width*net_height)*256
    return weight_mat

def train_network(epochs):
    global data, weight_mat, alpha, width
    #print (len(data))
    random.shuffle(data)
    #print (data)
    training = data[:len(data)-10]
    testing = data[len(training):]

    print(weight_mat[0])
    print ()
    for epoch in range(epochs):
        print("epoch:", epoch)
        eta = alpha * math.exp(-2*epoch/epochs)
        for example in range(len(training)):
            vec = training[random.randint(0, len(training)-1)]
            res = np.apply_along_axis(dist, 0, weight_mat, vec[0])
            _, _id = min((val, _id) for (_id, val) in enumerate(res))
            vec_data_diff = np.apply_along_axis(np.subtract, 0, weight_mat, vec[0])
            closest_loc = (_id%width, _id//width)
            dist_list = []
            for i in range(len(weight_mat[0])):
                loc = (i%width, i//width)
                dist_list.append(mexican_hat(closest_loc, loc, epoch/epochs))
            delta = -eta*(np.apply_along_axis(np.multiply, 1, vec_data_diff, dist_list))
            weight_mat += delta
    return training, testing

# calculates the distance between the weight and input vector
def dist(weight, vec):
    diff = weight - vec.T
    square = diff.dot(np.diag(diff))
    dist = np.sqrt(np.sum(square))
    return dist

def top_dist(c1, c2):
    global width, height
    diff_x = abs(c1[0]-c2[0])
    diff_y = abs(c1[1]-c2[1])
    diff_x = diff_x - width if (diff_x > width//2) else diff_x
    diff_y = diff_y - height if (diff_y > height//2) else diff_y

    return math.sqrt((diff_x)**2+(diff_y)**2)

def gaussian(loc1, loc2, r):
    return math.exp(-r * math.pow(top_dist(loc1, loc2),2))

def mexican_hat(loc1, loc2, r):
    return 4*math.exp(-r * math.pow(top_dist(loc1, loc2),2))-math.exp(-0.6*r * math.pow(top_dist(loc1, loc2),2))

def test_network(training, testing):
    global weight_mat, width, height
    from scipy.misc import imsave
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    graph = {}
    
    # calculate distances from all nodes to all nodes
    distances = np.zeros((len(weight_mat[0]), len(weight_mat[0])))
    for r in range(len(distances)):
        for c in range(r,len(distances[0])):
            d = dist(weight_mat[:,r], weight_mat[:,c])
            distances[r][c] = d
            distances[c][r] = d

    max_val = np.argmax(distances)
    distances = distances/max_val
    distances = 1-distances


    correct = np.zeros(width*height)
    incorrect = np.zeros(width*height)
    for vec in training:
        res = np.apply_along_axis(dist, 0, weight_mat, vec[0])
        _, _id = min((val, _id) for (_id, val) in enumerate(res))
        loc = (_id%width, _id//width)
        if vec[1] == 1:
            correct[loc[0]*width+loc[1]] += 1
        else:
            incorrect[loc[0]*width+loc[1]] += 1
    
    node_sum= correct+incorrect

    with np.errstate(divide='ignore', invalid='ignore'):
        incorrect_weight = np.true_divide(incorrect, node_sum)
        correct_weight = np.true_divide(correct, node_sum)
        #c = np.true_divide( a, b )
        incorrect_weight[ ~ np.isfinite( incorrect_weight )] = 0  # -inf inf NaN
        correct_weight[ ~ np.isfinite( correct_weight )] = 0
        incorrect_weight = np.reshape(incorrect_weight.dot(distances), (width, height))
        correct_weight = np.reshape(correct_weight.dot(distances), (width, height))

        print (np.true_divide(incorrect, node_sum))
        #print()
        #print(correct)


        rgb = np.dstack((128+(correct_weight-incorrect_weight)*128, 128+(correct_weight-incorrect_weight)*128, 128+(correct_weight-incorrect_weight)*128))
        print (rgb.shape)
        imsave("test_res6.png", rgb)
        

    '''
    if (loc not in graph):
        graph[loc] = (1, 1-vec[1])
    else:
        graph[loc] = (graph[loc][0]+1, (1-vec[1])+(graph[loc][1]))
    '''


    '''
    percent = np.zeros((width, height)).astype(float)
    for key in graph.keys():
        dict_val = graph[key]
        percent[key[0]][key[1]] = dict_val[1]/dict_val[0] if dict_val[1] > 0 else -1
    '''

def print_network_img():
    global weight_mat, width, height
    from scipy.misc import imsave
    img = np.zeros((width, height, 3))
    for i in range(width*height):
        x = i%width
        y = i//width
        img[x][y] = weight_mat[:, i]
    imsave("res4.jpg", img)


if __name__ == "__main__":
    #data = read_img_data()
    data = read_data(16)
    weight_mat = create_weight_mat(len(data[0][0]), width, height)
    training, testing = train_network(250)
    #print_network_img()
    test_network(training, testing)

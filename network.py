#Note: some code is borrowed from: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#also from https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/

import torch.nn
import torch.nn.functional
import torch.optim
import numpy.random
import math
import pandas
from sklearn.preprocessing import LabelEncoder

class Net(torch.nn.Module):

    def __init__(self, cat_cols, cont_cols, outputId, data):
        super(Net, self).__init__()
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.outputId = outputId
        
        #input: data row
        #compute number of embeddings
        embeddings = [data[col].nunique() for col in cat_cols]
        
        #Embed the categoricals
        self.embedLayer = torch.nn.ModuleList([torch.nn.Embedding(i, i * 2) for i in embeddings])
        
        #normalize the numericals
        self.bn_layer = torch.nn.BatchNorm1d(len(self.cont_cols))
        
        # Linear Layers
        self.fc1 = torch.nn.Linear(sum(embeddings) * 2 + len(cont_cols), 50)
        self.fc2 = torch.nn.Linear(50, 1)
    def forward(self, x):
        # Embedding Layer
        cat_encoded = [embedLayer(torch.tensor(x[self.cat_cols[i]].values, dtype=torch.int64)) for i,embedLayer in enumerate(self.embedLayer)]
        cat_encoded = torch.cat(cat_encoded, 1)
        cont_normalized = self.bn_layer(torch.tensor(x[cont_cols].values, dtype=torch.float32))
        x = torch.cat([cat_encoded, cont_normalized], 1)
        
        # Linear Layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.squeeze(x)

#todo date
#iffy - networkDomain
data = pandas.read_csv("train_clean.csv")
data["transactionRevenue"] = data["transactionRevenue"].apply(lambda x: math.log(1 + x))
cat_cols = ["channelGrouping", "browser", "operatingSystem", "isMobile", "deviceCategory", "continent", "subContinent", "country", "region", "city"]
cont_cols = ["visitNumber", "visitStartTime", "hits", "pageviews", "newVisits", "bounces"]
outputId = "fullVisitorId"
output = "transactionRevenue"
net = Net(cat_cols, cont_cols, outputId, data)
print(net)

#zero the gradient buffers
net.zero_grad()

#label encode the categorical variables
label_encoders = {}
for cat_col in cat_cols:
    label_encoders[cat_col] = LabelEncoder()
    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

#create testing and training set
msk = numpy.random.rand(len(data)) < 0.8
training_data = data[msk]
testing_data = data[~msk]
    
# create the optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

BATCH_SIZE = 100
# training loop
for epoch in range(5):
    order = numpy.random.choice(training_data.shape[0], size=training_data.shape[0], replace=False)
    for i in range(0, len(order) - BATCH_SIZE, BATCH_SIZE):
        miniBatch = training_data.iloc[order[i:i+BATCH_SIZE]]
        optimizer.zero_grad()   # zero the gradient buffers
        loss = criterion(net(miniBatch), torch.tensor(miniBatch[output].values, dtype=torch.float32))
        loss.backward()
        optimizer.step()    # Does the update

#testing loop
output = net(testing_data)

#save output
testing_data.loc[:,"Prediction"] = output.detach().tolist()
testing_data.loc[:,[outputId, "Prediction"]].to_csv("output.csv")
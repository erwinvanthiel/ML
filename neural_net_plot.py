import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(111)
torch.manual_seed(11)


# REGRESSION
# x = np.arange(100) / 10
# y = np.random.normal(0, 2, 100) + x
# w = np.dot(x,x) / np.dot(x,y)

# plt.scatter(x,y)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.plot(x,w*x, color='r')


# CLASSIFICATION

# Separated data
x1 = np.ones(20) * 1 + np.random.normal(0, 0.4, 20)
y1 = np.ones(20) * 4 + np.random.normal(0, 0.4, 20)

x2 = np.ones(20) * 4 + np.random.normal(0, 0.4, 20)
y2 = np.ones(20) * 1 + np.random.normal(0, 0.4, 20)

x3 = np.ones(20) * 1 + np.random.normal(0, 0.4, 20)
y3 = np.ones(20) * 1 + np.random.normal(0, 0.4, 20)

x4 = np.ones(20) * 4 + np.random.normal(0, 0.4, 20)
y4 = np.ones(20) * 4 + np.random.normal(0, 0.4, 20)


# Merged data 
x = np.concatenate((x1,x2,x3,x4), axis=0)
y = np.concatenate((y1,y2,y3,y4), axis=0)
t = np.concatenate((np.ones(20), np.ones(20),np.zeros(20), np.zeros(20)), axis=0)



data = torch.tensor(np.column_stack((x,y))).float()

# Model
model = torch.nn.Sequential(
					  torch.nn.Linear(2, 2),
					  torch.nn.Sigmoid(),
					  torch.nn.Linear(2,1),
					  torch.nn.Sigmoid()
					  )
# Optimizer and Loss
def loss_function(pred, target):
	error = (target - pred) * (target - pred)
	return torch.sum(error) 

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train
for i in range(10000):
	model.zero_grad()
	pred = model(data).flatten()
	loss = loss_function(pred,torch.tensor(t).float())
	loss.backward()
	optimizer.step()
	print(loss)


plt.scatter(x1,y1, color='red', label='Bad diet')
plt.scatter(x2,y2, color='red')
plt.scatter(x3,y3, color='green', label='Good diet')
plt.scatter(x4,y4, color='green')
plt.xticks([])
plt.yticks([])
plt.xlabel("Body weight")
plt.ylabel("Time spent lifting weights")
plt.legend()
plt.show()

decision_points_x = []
decision_points_y = []

for i in range(100):
	for j in range(100):
		decision_points_x.append(j/17)
		decision_points_y.append(i/17)

plot_data = torch.tensor(np.column_stack((np.array(decision_points_x),np.array(decision_points_y)))).float()
pred = np.array(list(np.where(model(plot_data) > 0.5, 1,0).flatten()))
x_class1 = np.array(decision_points_x)[pred == 0]
y_class1 = np.array(decision_points_y)[pred == 0]
x_class2 = np.array(decision_points_x)[pred == 1]
y_class2 = np.array(decision_points_y)[pred == 1]

plt.xticks([])
plt.yticks([])
plt.scatter(x_class1, y_class1, color="green", alpha=0.05)
plt.scatter(x_class2, y_class2, color="red", alpha=0.05)
plt.xlabel("Body weight")
plt.ylabel("Time spent lifting weights")

plt.scatter(x1,y1, color='red')
plt.scatter(x2,y2, color='red', label="Bad diet",)
plt.scatter(x3,y3, color='green', label="Good diet")
plt.scatter(x4,y4, color='green')
plt.show()

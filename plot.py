import matplotlib.pyplot as plt
import numpy as np
import torch
np.random.seed(111)


# REGRESSION
x = np.linspace(50,200,100)
y = np.random.normal(0, 15, 100) + 0.5*x
w = np.dot(x,y) / np.dot(x,x)
print("w = ",w)
# w = 0.45

# Data points
px = [142.46, 142.46]
py = [38.2, 64.3]

# Create the plot
plt.plot(px, py, linestyle='dotted', color='black')

# Add the label
plt.text(160, 50, '} error', ha='right')


plt.scatter(x,y)
plt.xlabel("Height(cm)")
plt.ylabel("Weight(Kg)")
plt.plot(x,w*x, color='r')
plt.show()


# # CLASSIFICATION

# # Separated data
# x1 = np.ones(20) * 1 + np.random.normal(0, 0.5, 20)
# y1 = np.ones(20) * 2 + np.random.normal(0, 0.5, 20)

# x2 = np.ones(20) * 2 + np.random.normal(0, 0.5, 20)
# y2 = np.ones(20) * 1 + np.random.normal(0, 0.5, 20)


# # Merged data 
# x = np.concatenate((x1,x2), axis=0)
# y = np.concatenate((y1,y2), axis=0)
# t = np.concatenate((np.ones(20), -1*np.ones(20)), axis=0)

# data = torch.tensor(np.column_stack((x,y))).float()

# # Model
# model = torch.nn.Sequential(
# 					  torch.nn.Linear(2, 1)
# 					  )
# # model.state_dict()["0.weight"][0][0] = torch.tensor(-1.5)
# # model.state_dict()["0.weight"][0][1] = torch.tensor(-1.5)
# # model.state_dict()["0.bias"][0] = torch.tensor(3)

# # Optimizer and Loss
# def loss_function(pred, target):
# 	loss = torch.where(target * pred < 0, torch.abs(pred), torch.zeros_like(pred))
# 	return torch.sum(loss)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0075)

# for i in range(200):
# 	w1 = model.state_dict()["0.weight"][0][0].item()
# 	w2 = model.state_dict()["0.weight"][0][1].item()
# 	b = model.state_dict()["0.bias"][0]
# 	decision_boundary_x = np.linspace(0,3,1000)
# 	decision_boundary_y =  (-b / w2) - (w1/w2) * decision_boundary_x

# 	with torch.no_grad():
# 		plt.figure()
# 		plt.scatter(x1,y1, color='green', label='not a developer')
# 		plt.scatter(x2,y2, color='red', label='developer')
# 		plt.plot(decision_boundary_x, decision_boundary_y)
# 		plt.xlabel("Cups of coffee per day")
# 		plt.ylabel("Hours of physical activity per day")
# 		plt.legend()
# 		plt.xlim(0, 3)  # Set the x-axis limits from 0 to 6
# 		plt.ylim(0, 3)
# 		plt.savefig('gif/my_plot{0}.png'.format(i))
# 		# plt.show()

# 	model.zero_grad()
# 	pred = model(data).flatten()
# 	loss = loss_function(pred,torch.tensor(t).float())
# 	loss.backward()
# 	optimizer.step()
# 	print(loss, i)




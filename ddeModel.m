function dy = ddeModel(y,theta)
% y is the historical states
% dy = theta.fc2.Weights*tanh(theta.fc1.Weights*y);
% dy = theta.fc2.Weights*tanh(theta.fc1.Weights*y+theta.fc1.Bias);
dy = theta.fc3.Weights*tanh(theta.fc2.Weights*tanh(theta.fc1.Weights*y+theta.fc1.Bias)+theta.fc2.Bias);
end
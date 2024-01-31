function [par_grad,x0_grad,loss] = gradDeriv(dlx,neuralOdeParameters,dx_targets)
% dlx is the picked-out delayed state
dlxdot = ddeModel(dlx,neuralOdeParameters);
loss = l1loss(dlxdot,dx_targets,'NormalizationFactor','all-elements','DataFormat','CBT');
par_grad = dlgradient(loss,neuralOdeParameters);
x0_grad = dlgradient(loss,dlx);
end
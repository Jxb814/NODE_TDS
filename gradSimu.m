function [par_grad,x0_grad,loss] = gradSimu(tspan,dlX0,neuralOdeParameters,targets,BaseNet)
% dlx is the picked-out delayed state
Xall = dlode45(@(t,y,theta)odeModel(t,y,theta,BaseNet),tspan,dlX0,neuralOdeParameters,'DataFormat','CB');
dlX = Xall(1,:,:);  % last dimension is the steps
loss = l1loss(dlX,targets(1,:,:),'NormalizationFactor','all-elements','DataFormat','CBT');
par_grad = dlgradient(loss,neuralOdeParameters);
x0_grad = dlgradient(loss,dlX0);
end

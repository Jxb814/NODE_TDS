clear
clc
close all
plottingPreferences()
%% ode simulation data, drop the transient part
tau = 1;
beta = 4;
n = 9.65;
gamma = 2;
trueModel = @(t,x,xdelay) beta*xdelay/(1+xdelay^n)-gamma*x;
tr_c =linspace(0.5,1.5,100); % 100 sets

T = 20;
dt = 0.05;
tspan = [0 T];
N = round(T/dt)+1;
Tst = 10; % drop first 10 second 
T_tr = 7;
N_tr = round(T_tr/dt)+1; % take fisrt part of remaining data for training

tau_max = 1.5;
nX = round(tau_max/dt)+1;  % number of states
Xtrain_all = {};

BaseNet0 = struct;
BaseNet0.DiffMat = diffMat(nX,dt);
BaseNet0.dt = dt;

% generate data + derivative
for k = 1:length(tr_c)
    hist = @(t) tr_c(k);
    sol = dde23(trueModel,tau,hist,tspan);
    tint = linspace(0,T,N);
    yint{k} = deval(sol,tint);

    tint_tr = linspace(Tst,Tst+T_tr,N_tr);
    yint_tr{k} = yint{k}(round(Tst/dt):round(Tst/dt)+N_tr-1);

    % obtain derivative via central difference
    dy{k} = ctrDiff(yint{k},tint);
    dy_tr{k} = ctrDiff(yint_tr{k},tint_tr);

    m = length(tint_tr)-nX+1;  % number of datapoints
    % the first dim of xTrain should n*n_x if it is not a scalar sys
    xTrain = zeros(nX,m);
    dxTrain = zeros(nX,m);
    for i = 1:m
        xTrain(:,i) = yint_tr{k}(i:nX+i-1);
        dxTrain(:,i) = dy_tr{k}(i:nX+i-1);
    end
    xTrain=flipud(xTrain);
    Xtrain_all{k}=xTrain;
    x0{k} = xTrain(:,1);
    dxTrain=flipud(dxTrain);
    dXtrain_all{k}=dxTrain;
end
t = tint_tr(1:m);
t_plot_tr = zeros(nX,m-1);
for i = 1:nX
    t_plot_tr(end-i+1,:) = tint_tr(1+i:i+m-1);
end
%%
figure(1)
set(gcf,'Position',[300 100 600 250])
hold on
for k = 1 %1:100:length(tr_c)
    % plot all data dde23
    plot(tint,yint{k}, 'r--');
    plot([Tst Tst],[0.3 1.4], 'k-');
    plot([Tst+T_tr Tst+T_tr],[0.3 1.4], 'k-');
    legend('data',Location='best')
end
hold off
box on;
xlabel('$t$')
ylabel('$x$')
ylim([0.3 1.4])

%% use dde simulation for training
steps_pre1 = 30; % simulation steps m-1
learnRate1 = 0.01;
numIter = 1000;
steps_pre2 = 10;
learnRate2 = learnRate1;

ratio = 1;
plotFrequency = 10;
gradDecay = 0.9;
sqGradDecay = 0.999;
miniBatchSize = 10; % miniBatchSize <= m-neuralOdeTimesteps

hiddenSize = 5;
stateSize  = size(Xtrain_all{1},1);

% delay parameter
Tau = dlarray(tau_max*rand([1,3]));
% Tau = dlarray([0 tau_max*rand([1,2])]);
% Tau = dlarray([0 tau_max*rand(1)]);
% Tau = dlarray([0.975, 1.475]);
% Tau = dlarray(0:0.05:tau_max);
Tau = min(max(0,Tau),tau_max-0.01*dt);
BaseNet = struct;
BaseNet.DiffMat = diffMat(stateSize,dt);
BaseNet.delayPick =  delayPick(Tau,dt,stateSize);
BaseNet.dt = dt;
delaySize = size(BaseNet.delayPick,1);
% store the parameters for NODE
NODE = struct;
NODE.fc1 = struct;
sz = [hiddenSize delaySize];
NODE.fc1.Weights = initializeGlorot(sz, hiddenSize, delaySize);
NODE.fc1.Bias    = initializeZeros([hiddenSize 1]);

NODE.fc2 = struct;
sz = [hiddenSize hiddenSize];
NODE.fc2.Weights = initializeGlorot(sz, hiddenSize, hiddenSize);
NODE.fc2.Bias    = initializeZeros([hiddenSize 1]);

NODE.fc3 = struct;
sz = [1 hiddenSize];
NODE.fc3.Weights = initializeGlorot(sz, 1, hiddenSize);

start = tic;
Loss = zeros(1,numIter);
Tau_tr = zeros(delaySize,numIter);

aveGrad   = [];
aveSqGrad = [];
aveGrad_tau = [];
aveSqGrad_tau = [];
steps_max = m;
timesteps = (0:steps_pre1)*dt;

%%
for iter=1:numIter
    % Create batch 
    [dlx0, targets, dx, dx_targets] = createMiniBatch(steps_max, steps_pre1, miniBatchSize, Xtrain_all, dXtrain_all);

%   % simulation loss
    [grads,dx_grads,loss] = dlfeval(@gradSimu,timesteps,dlx0,NODE,targets,BaseNet);
    grads_tau = -sum(BaseNet.delayPick*(dx_grads.*dx),2)';    

%   % derivative loss
%     dlx_pick = BaseNet.delayPick*dlx0; % take out relavent states
%     xdot_pick = BaseNet.delayPick*dx;
%     [grads,dx_grads,loss] = dlfeval(@gradDeriv,dlx_pick,NODE,dx_targets);
%     grads_tau = -sum(dx_grads.*xdot_pick,2)';
    
    Tau_tr(:,iter)=extractdata(Tau');
    currentLoss = double(extractdata(loss));
    Loss(iter) = currentLoss;
    
    if mod(iter,plotFrequency) == 0  || iter == 1
        figure(2)
        set(gcf,'Position',[50 50 600 700])
        clf
        subplot(3,1,1)
        semilogy(1:iter, Loss(1:iter))
        xlabel('Iteration')
        ylabel('Loss')
  
        subplot(3,1,2)
        % Use ode45 to compute the solution 
        % plotting is slow bc it contains simulations
        y = dlode45(@(t,y,theta)odeModel(t,y,theta,BaseNet),t,dlarray(x0{1}),NODE,'DataFormat','CB');
        y = extractdata(y);
        hold on
        plot(tint_tr(2:end),yint_tr{1}(2:end),'r--', 'LineWidth',2)
        plot(t_plot_tr,squeeze(y),'b-')
        xlabel('$t$')
        ylabel('$x$')
        hold off
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        box on
        title("Iter = "+iter+ ", loss = " +num2str(currentLoss)+", Elapsed: "+string(D))
        legend('Training ground truth', 'Predicted')

        subplot(3,1,3)
        hold on
        for k = 1:delaySize
            plot(1:iter, Tau_tr(k,1:iter),'LineWidth',2)
        end
        plot(1:iter+1, ones(1,iter+1), 'k--','LineWidth',2)
        plot(1:iter+1, zeros(1,iter+1),'k--','LineWidth',2)
        ylim([0,1.5])
        xlim([1,iter+1])
        xlabel('Iteration')
        ylabel('delays')
        hold off
        box on
        title('Learned delays')
    end
    % Update network
    upd.Tau = Tau;
    upd.NODE = NODE;
    Grad.Tau = grads_tau;
    Grad.Node = grads;
    [upd,aveGrad,aveSqGrad] = adamupdate(upd,Grad,aveGrad,aveSqGrad,iter,...
        ratio*learnRate1,gradDecay,sqGradDecay);
    Tau = upd.Tau;
    NODE = upd.NODE;
    Tau = min(max(0,Tau),tau_max-0.01*dt);

    BaseNet.delayPick =  delayPick(Tau,dt,stateSize);
end

%%
figure(3)
set(gcf,'Position',[100 100 500 300])
semilogy(1:numIter, Loss(1:numIter))
xlabel('Iteration')
ylabel('Loss')
grid on
title(['Loss vs. iterations '])

figure(4)
set(gcf,'Position',[100 100 500 300])
hold on
for k = 1:delaySize
    plot(1:numIter, Tau_tr(k,1:numIter),'LineWidth',2)
end
plot(1:numIter+1, ones(1,numIter+1), 'k--','LineWidth',2)
plot(1:numIter+1, zeros(1,numIter+1),'k--','LineWidth',2)
ylim([0,tau_max])
xlim([1,numIter+1])
xlabel('Iteration')
ylabel('delays')
hold off
box on
title('Learned delays')

%% simulation (ode45 simulation with initial state)
% training + testing
tt = Tst:dt:T-tau_max;
tt_ext = Tst:dt:T;
t_plot = zeros(nX,length(tt));
for i = 1:nX
    t_plot(i,:) = tint(end-length(tt)-i+2:end-i+1);
end
y_ode = cell(1,100);
y_nn = cell(1,100);
y_nn_ave = cell(1,100);
y_ode_ave=cell(1,100);

for k =  1:length(tr_c)
    [tsim,y_ode{k}] = ode45(@(t,y)odeModel_gt(t,y,BaseNet),tt,x0{k});
    [~,y_nn{k}] = ode45(@(t,y)odeModel(t,y,NODE,BaseNet),tt,x0{k});
    y_nn_ave{k} = zeros(size(tt_ext));
    y_ode_ave{k} = zeros(size(tt_ext));
    for kk = 1:length(tt_ext)
        y_nn_ave{k}(kk) = mean(diag(y_nn{k}',-nX+kk));
        y_ode_ave{k}(kk) = mean(diag(y_ode{k}',-nX+kk));
    end
end

% only testing simulation (error does not accumulate from training)
tt_ts = Tst+T_tr-tau_max:dt:T;
tt_ext_ts = Tst+T_tr-tau_max:dt:T;
t_plot_ts = zeros(nX,length(tt_ts));
for i = 1:nX
    t_plot_ts(i,:) = tint(end-length(tt_ts)-i+2:end-i+1);
end
y_ode_ts = cell(1,100);
y_nn_ts = cell(1,100);
y_nn_ave_ts = cell(1,100);
y_ode_ave_ts=cell(1,100);
for k =  1:length(tr_c)
    [tsim,y_ode_ts{k}] = ode45(@(t,y)odeModel_gt(t,y,BaseNet),tt_ts,Xtrain_all{k}(:,end));
    [~,y_nn_ts{k}] = ode45(@(t,y)odeModel(t,y,NODE,BaseNet),tt_ts,Xtrain_all{k}(:,end));
    y_nn_ave_ts{k} = zeros(size(tt_ext_ts));
    y_ode_ave_ts{k} = zeros(size(tt_ext_ts));
    for kk = 1:length(tt_ext_ts)
        y_nn_ave_ts{k}(kk) = mean(diag(y_nn_ts{k}',-nX+kk));
        y_ode_ave_ts{k}(kk) = mean(diag(y_ode_ts{k}',-nX+kk));
    end
end

%%
k = 1;
figure(102)
set(gcf,'Position',[300 100 700 300])
hold on
% plot the average
plot(tt_ext(1:N_tr),yint{k}(round(Tst/dt):round(Tst/dt)+N_tr-1),'r--')
plot(tt_ext(1:N_tr),y_ode_ave{k}(1:N_tr),'k--')
plot(tt_ext(1:N_tr),y_nn_ave{k}(1:N_tr),'b-')
legend('ground truth','ODE simu','NODE simu')
title('training')
box on;
grid on;

figure(104)
set(gcf,'Position',[300 100 700 300])
hold on
% plot the average
plot(tt_ext_ts,yint{k}(round((Tst+T_tr-tau_max)/dt):end-1),'r--')
plot(tt_ext_ts,y_ode_ave_ts{k},'k--')
plot(tt_ext_ts,y_nn_ave_ts{k},'b-')
legend('ground truth','ODE simu','NODE simu')
title('testing only')
box on;
grid on;

%% phase plot
figure(103)
set(gcf,'Position',[100 100 300 300])
hold on
for k = 1:10:length(tr_c)
    plot(yint{k}(round(Tst/dt):round(Tst/dt)+N_tr-21),yint{k}(round(Tst/dt)+20:round(Tst/dt)+N_tr-1),'r--')
    plot(y_nn_ave{k}(1:N_tr-20),y_nn_ave{k}(21:N_tr),'b-')
end
hold off
ylabel('$x(t)$')
xlabel('$x(t-\tau )$')
title('10 training trajectories')
legend('Data','Prediction','location','best')
box on;
axis equal

figure(105)
set(gcf,'Position',[100 100 300 300])
hold on
for k = 1:10:length(tr_c)
    plot(yint{k}(round((Tst+T_tr)/dt):end-21),yint{k}(round((Tst+T_tr)/dt)+20:end-1),'r--')
    plot(y_nn_ave_ts{k}(nX:end-20),y_nn_ave_ts{k}(20+nX:end),'b-')
end
hold off
ylabel('$x(t)$')
xlabel('$x(t-\tau )$')
title('10 testing trajectories')
box on;
axis equal

filename = ['MG_L2N5_iter',num2str(numIter),'_step',num2str(steps_pre1),'_simu'];
save(filename)

%% define the ground truth model in ode for plotting
function dy = odeModel_gt(~,y,BaseNet)
% ground truth of the MG model in ODE
dy = BaseNet.DiffMat*y;
tau = 1;
k = round(tau/BaseNet.dt)+1;
dy(1,:) = 4*y(k)/(1+y(k)^9.65)-2*y(1);
end


function Mat = diffMat(stateSize,dt)
% central difference
Mat1 = diag(-0.5/dt*ones(1,stateSize-1),1);
Mat2 = diag(0.5/dt*ones(1,(stateSize-1)),-1);
Mat = Mat1+Mat2;
Mat(1,2)=0;
Mat(end,end-1)=1/dt;
Mat(end,end)=-1/dt;
end
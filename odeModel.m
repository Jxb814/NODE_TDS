function dy = odeModel(~,y,theta,BaseNet)
dy = BaseNet.DiffMat*y;
dy(1,:) = ddeModel(BaseNet.delayPick*y,theta);
end
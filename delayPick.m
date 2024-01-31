function D = delayPick(T,dt,stateSize)
% % round the delays to the closest integer multiple
% delayPos = round(T/dt)+1;
% delaySize = length(T); % two terms including non-delay
% D = zeros(delaySize,stateSize);
% for k = 1:delaySize
%     D(k,delayPos(k))=1;
% end
% interpolate the delays
delayPos = floor(T/dt);
alpha = T/dt-delayPos;
delaySize = length(T);
D = zeros(delaySize,stateSize);
for k = 1:delaySize
    D(k,delayPos(k)+1)=(1-alpha(k));
    D(k,delayPos(k)+2)=alpha(k);
end
end
%% fuzzy gains
kp = 1.18;
ki = 16.16;

a = kp/ki;

% Ke = 1;
% K = kp/F_function(a,Ke,1);
% Kd=a*Ke;
% Ts = 0.01;

% after tunning
Ke = 1.5;
K = 24;
Kd = a*Ke;
    

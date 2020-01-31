%% initiall values ,changed after tunning
%a=0.073
%K==16.16
%Kd=0.073
%Ke=1

%%after tunning values
a = 0.0730;
K = 24;
Ke = 1.5;
Kd =a*Ke;
Ts = 0.01;


input = timeseries( 150.*trapmf([0:1:30],[0 10 20 30]) );   
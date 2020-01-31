%%pre tunning values
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
    
input = timeseries([ones(10,1)*150;ones(10,1)*100;ones(10,1)*150]);   



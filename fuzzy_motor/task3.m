a = 0.073;
K =10;
Ke = 0.2;
Kd = a*Ke;
Ts = 0.01;

input = timeseries([ones(100,1).*150]);   
TL = timeseries([ones(10,1).*0.5*100 ; ones(10,1)*100 ; ones(80,1).*0.5*100]);   

%% use them AFTER you run simulation in simulink
time = ScopeData(:,1);
ustep = ScopeData(:,2);
pi = ScopeData(:,3);
plot(ustep)
hold on
plot(pi)
hold off
legend('input' , 'system response')
xlabel('time in milliseconds')
ylabel('volt')
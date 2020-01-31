%% Data Export from Simulink to MATLAB

% YPATIA PAIKSE ME TA NOUMERA TOU SIMULINK
% Den to exw kalibrarei akoma kai prepei na ftiakse kai check gia to
% response time na mas leei an einai ok i oxi

close all
sim('Ypatia1')

commanded_velocity = max(motor_command{1}.Values);                                  %commaded velocity [rad/s], [rpm], ...
motor_velocity_out = yout{1}.Values.Data;                                           %motor velocity output [rad/s], [rpm], ...
examined_time = yout{1}.Values.Time;                                                %time [s]
error = (motor_velocity_out-commanded_velocity) / commanded_velocity*100;           %error in regarding the commanded velocity [%]

overshoot = error(find(motor_velocity_out== max(motor_velocity_out)));              %overshoot [%]

time_delay = examined_time( min(find(motor_command{1}.Values.Data ~= 0)));          %time delay for motor command--> defined in simulink model [s]
response_time = examined_time( min(find(abs(error) < 32))) - time_delay;            %the response time is defined as the time where the Y>= 68% of the commanded Y [s]


%% Voltage check
v_in = voltage_input{1}.Values.Data;                                                %signal after the PI--> equivalent to voltage [V]  
voltage_limit = 200;                                                                 %voltage limit defined at the problems specifications [V]
if sum(v_in > voltage_limit)
    fprintf('ATTENTION: There is a voltage overpass \n')
else
    fprintf('PI commanded voltage is OK \n')
end
    
%% Overshoot check
accepted_overshoot = 5;                                                               %accepted overshoot [%]
if overshoot > accepted_overshoot
    fprintf('Oveshoot NOK \n')
else
    fprintf('Oveshoot OK \n')
end

%% Response time check
accepted_response_time = 160e-3;                                                      %response time in [s]
if response_time <= accepted_response_time
    fprintf('Response time is OK \n')
else
    fprintf('Response time is NOK \n')
end
%% Plotter
figure()
plot(examined_time, motor_velocity_out)
hold on
ylabel ('[rad/s]')
yyaxis right
plot (examined_time, error)
ylabel ('[%]')
xlabel('time [s]')
legend ('Motor Velocity', 'Relative Error')
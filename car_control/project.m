 

%% problem domain
% y in [0,4] , x in [0 , 10]
Ts = 0.1;
iter = 200/Ts;
pos = zeros(iter,2);

%% specify desired position
xd = 10;
yd = 3.2;

%% specify starting position
% x = 4.1;
% y = 0.3;
pos(1,1) = 4.1;
pos(1,2) = 0.3;
theta = -90;

%% specify velocity
v = 0.05;                                  % pixel per iteration 

%% create FLC 
fis = readfis('car_control.fis');


%% simulate car's movement
K = 1;
for i=2:iter
    [ dh , dv ] = estimate_distance(pos(i-1,1),pos(i-1,2));
    dh = abs(dh/10);
    dv = abs(dv/4);
    theta = normalize_theta(theta);
    theta = theta + K*evalfis([dv, dh, theta],fis);
    pos(i,:) =  update_position(pos(i-1,:),v,theta,Ts);
end

plot(pos(:,1),pos(:,2))
hold on
xy = [5 0;5 0.5 ; 5 1; 5.5 1; 6 1; 6 1.5 ; 6 2; 6 2; 6.5 2; 7 2; 7 3;8 3; 10 3];
plot(xy(:,1),xy(:,2))
plot(xd.*[1 1 1 1 1],[0 1 2 3 4]);
plot(xd,yd,'*');
hold off
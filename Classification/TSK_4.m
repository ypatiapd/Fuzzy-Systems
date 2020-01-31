%%Classification problem 
% firts task- simple dataset
%ypatia dami

%% LOAD AND SPLIT DATA
%load aviila-num.txt from import data button
data=aviilanum;
% Keep the data of different outputs to separate arrays
val1 = data(data(:, end) == 0, :);
val2 = data(data(:, end) == 1, :);
val3 = data(data(:, end) == 3, :);
val4 = data(data(:, end) == 4, :);
val5 = data(data(:, end) == 5, :);
val6 = data(data(:, end) == 6, :);
val7 = data(data(:, end) == 7, :);
val8 = data(data(:, end) == 8, :);
val9 = data(data(:, end) == 9, :);
val10 = data(data(:, end) == 10, :);
val11= data(data(:, end) == 11, :);
val12= data(data(:, end) == 12, :);

% Flags for the index of separating between the sets
first_split_one = round(0.6 * length(val1));
second_split_one = round(0.8 * length(val1));

first_split_two = round(0.6 * length(val2));
second_split_two = round(0.8 * length(val2));

first_split_three = round(0.6 * length(val3));
second_split_three = round(0.8 * length(val3));

first_split_four = round(0.6 * length(val4));
second_split_four = round(0.8 * length(val4));

first_split_five = round(0.6 * length(val5));
second_split_five = round(0.8 * length(val5));

first_split_six = round(0.6 * length(val6));
second_split_six = round(0.8 * length(val6));

first_split_seven = round(0.6 * length(val7));
second_split_seven = round(0.8 * length(val7));

first_split_eight = round(0.6 * length(val8));
second_split_eight = round(0.8 * length(val8));

first_split_nine = round(0.6 * length(val9));
second_split_nine = round(0.8 * length(val9));

first_split_ten = round(0.6 * length(val10));
second_split_ten = round(0.8 * length(val10));

first_split_eleven = round(0.6 * length(val11));
second_split_eleven = round(0.8 * length(val11));

first_split_twelve = round(0.6 * length(val12));
second_split_twelve = round(0.8 * length(val12));

% 60% for training, 20% for validation, 20% for checking
training_data = [val1(1:first_split_one, :); val2(1:first_split_two, :); val3(1:first_split_three,:) ;val4(1:first_split_four, :); val5(1:first_split_five, :); val6(1:first_split_six, :);val7(1:first_split_seven, :); val8(1:first_split_eight, :); val9(1:first_split_nine,:);val10(1:first_split_ten, :); val11(1:first_split_eleven, :); val12(1:first_split_twelve,:)];
evaluation_data = [val1(first_split_one + 1:second_split_one, :); val2(first_split_two + 1:second_split_two, :); val3(first_split_three + 1:second_split_three, :);val4(first_split_four + 1:second_split_four, :); val5(first_split_five + 1:second_split_five, :); val6(first_split_six + 1:second_split_six, :);val7(first_split_seven + 1:second_split_seven, :); val8(first_split_eight + 1:second_split_eight, :); val9(first_split_nine + 1:second_split_nine, :);val10(first_split_ten + 1:second_split_ten, :); val11(first_split_eleven + 1:second_split_eleven, :); val12(first_split_twelve + 1:second_split_twelve, :)];
testing_data = [val1(second_split_one + 1:end, :); val2(second_split_two + 1:end, :); val3(second_split_three + 1:end, :);val4(second_split_four + 1:end, :); val5(second_split_five + 1:end, :); val6(second_split_six + 1:end, :);val7(second_split_seven + 1:end, :); val8(second_split_eight + 1:end, :); val9(second_split_nine + 1:end, :);val10(second_split_ten + 1:end, :); val11(second_split_eleven + 1:end, :); val12(second_split_twelve + 1:end, :)];

% Shuffle the data
shuffled_data = zeros(size(training_data));
rand_pos = randperm(length(training_data));
for k = 1 : length(training_data)
    shuffled_data(k, :) = training_data(rand_pos(k), :);
end
training_data = shuffled_data;

shuffled_data = zeros(size(evaluation_data));
rand_pos = randperm(length(evaluation_data));
% new array
for k = 1 : length(evaluation_data)
    shuffled_data(k, :) = evaluation_data(rand_pos(k), :);
end
evaluation_data = shuffled_data;

shuffled_data = zeros(size(testing_data));
rand_pos = randperm(length(testing_data));
% new array
for k = 1 : length(testing_data)
    shuffled_data(k, :) = testing_data(rand_pos(k), :);
end
testing_data = shuffled_data;

%% TRAIN THE MODEL



 genfis_opt = genfisOptions('SubtractiveClustering');
 % 5rules
genfis_opt.ClusterInfluenceRange=[0.6 0.3 0.8 0.2 0.5 0.2 0.2 0.7 0.2 0.9 0.2];
initial_fis = genfis(training_data(:,1:10),training_data(:,11),genfis_opt);

%plot initial member functions
figure;
[x,mf] = plotmf(initial_fis,'input',1);
subplot(2,2,1);
plot(x,mf);

title('input1 ');
ylabel('Membership function');
saveas(gcf,'TSK_4_input1_initMF.png');

[x,mf] = plotmf(initial_fis,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 ');
ylabel('Membership function');
saveas(gcf,'TSK_4_input2_initMF.png');

[x,mf] = plotmf(initial_fis,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3');
ylabel('Membership function');
saveas(gcf,'TSK_4_input3_initMF.png');

[x,mf] = plotmf(initial_fis,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input4');
ylabel('Membership function');
saveas(gcf,'TSK_4_input4_initMF.png');


anfis_opt=anfisOptions('InitialFIS',initial_fis);
anfis_opt.ValidationData=[evaluation_data(:,1:10) evaluation_data(:,11)];
anfis_opt.EpochNumber=400;
[train_fis,trainError,stepSize,chkFIS,chkError] = anfis([training_data(:,1:10) training_data(:,11)],anfis_opt);


system_output = evalfis(testing_data(:,1:10),chkFIS);
system_output=round(system_output);
system_output(system_output < 1) = 1;
system_output(system_output > 12) = 12;
%plotting final mfs to compare with initial

figure;
[x,mf] = plotmf(chkFIS,'input',1);
subplot(2,2,1);
plot(x,mf);
title('input1');
ylabel('Membership function');
saveas(gcf,'TSK_4_input1_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 ');
ylabel('Membership function');
saveas(gcf,'TSK_4_input2_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3 ');
ylabel('Membership function');
saveas(gcf,'TSK_4_input3_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input3 ');
ylabel('Membership function');
saveas(gcf,'TSK_4_input4_finalMF.png');

% %% MODEL EVALUATION
%error matrix
N=length(testing_data);
error_matrix = confusionmat(testing_data(:,end), system_output)

%overal accuracy
correct=zeros(1,12)
for i=1: 12
    correct(i)=error_matrix(i,i);
end

overal_accuracy=sum(correct)/N;
%user and producer accuracy

PA=zeros(1,12);
UA=zeros(1,12);
for i=1 : 12
    PA(i)=error_matrix(i,i)/length(testing_data(testing_data(:,end)==i));
    UA(i)=error_matrix(i,i)/length(system_output(system_output==i));
end

%k metric

sum1=zeros(1,12);
sum2=zeros(1,12);

for i=1:12
    correct(i)=error_matrix(i,i);
    mul1(i)= length(system_output(system_output==i))*length(testing_data(testing_data(:,end)==i));
    mul2(i)= length(testing_data(testing_data(:,end)==i))*length(system_output(system_output==i));
end

K=(N*sum(correct)-sum(mul1))/(N^2-sum(mul2));

% 

fprintf('?? = %f PA = %f UA = %f K = %f \n',overal_accuracy, PA, UA, K );%% PLOTTING RESULTS



%fuzzy system results

figure;

plot(1:length(system_output),system_output,'r*',1:length(system_output),testing_data(1:length(system_output),11),'b*');
title('Fuzzy System output');
legend('Fuzzy System Output','Testing data Output');
saveas(gcf,'FuzzyOutput_TSK_4.png');

%training and evaluating errors

figure;

plot(1:length(trainError),trainError,1:length(trainError),chkError);
title('Learning Curves');
legend('Training Error', 'Testing Error');
saveas(gcf,'TrainingErrors_TSK_4.png');
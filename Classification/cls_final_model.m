%% High dimensionality dataset-Final model
%% Load and split data

format short;
data=zeros(7797,618);
data=readtable('isolet_csv.csv'); %read the data from excel sheet
data=table2array(data);

val1 = data(data(:, end) == 1, :);
val2 = data(data(:, end) == 2, :);
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
val13 = data(data(:, end) == 13, :);
val14 = data(data(:, end) == 14, :);
val15 = data(data(:, end) == 15, :);
val16 = data(data(:, end) == 16, :);
val17 = data(data(:, end) == 17, :);
val18 = data(data(:, end) == 18, :);
val19 = data(data(:, end) == 19, :);
val20 = data(data(:, end) == 20, :);
val21 = data(data(:, end) == 21, :);
val22 = data(data(:, end) == 22, :);
val23= data(data(:, end) == 23, :);
val24= data(data(:, end) == 24, :);
val25= data(data(:, end) == 25, :);
val26= data(data(:, end) == 26, :);

first_split_one = round(0.6 * size(val1,1));
second_split_one = round(0.8 * size(val1,1));

first_split_two = round(0.6 * size(val12,1));
second_split_two = round(0.8 * size(val2,1));

first_split_three = round(0.6 * size(val3,1));
second_split_three = round(0.8 * size(val3,1));

first_split_four = round(0.6 * size(val4,1));
second_split_four = round(0.8 * size(val4,1));

first_split_five = round(0.6 * size(val5,1));
second_split_five = round(0.8 * size(val5,1));

first_split_six = round(0.6 * size(val6,1));
second_split_six = round(0.8 * size(val6,1));

first_split_seven = round(0.6 * size(val7,1));
second_split_seven = round(0.8 * size(val7,1));

first_split_eight = round(0.6 * size(val8,1));
second_split_eight = round(0.8 * size(val8,1));

first_split_nine = round(0.6 * size(val9,1));
second_split_nine = round(0.8 * size(val9,1));

first_split_ten = round(0.6 *size(val10,1));
second_split_ten = round(0.8 *size(val10,1));

first_split_eleven = round(0.6 * size(val11,1));
second_split_eleven = round(0.8 * size(val11,1));

first_split_twelve = round(0.6 * size(val12,1));
second_split_twelve = round(0.8 * size(val12,1));

first_split_thirteen = round(0.6 * size(val13,1));
second_split_thirteen = round(0.8 * size(val13,1));

first_split_fourteen = round(0.6 *size(val14,1));
second_split_fourteen = round(0.8 * size(val14,1));

first_split_fifteen = round(0.6 * size(val15,1));
second_split_fifteen = round(0.8 *size(val15,1));

first_split_sixteen = round(0.6 * size(val16,1));
second_split_sixteen = round(0.8 * size(val16,1));

first_split_seventeen = round(0.6 * size(val17,1));
second_split_seventeen = round(0.8 * size(val17,1));

first_split_eighteen = round(0.6 * size(val18,1));
second_split_eighteen = round(0.8 *size(val18,1));

first_split_nineteen = round(0.6 *size(val19,1));
second_split_nineteen = round(0.8 * size(val19,1));

first_split_twenty = round(0.6 * size(val20,1));
second_split_twenty= round(0.8 * size(val20,1));

first_split_twentyone = round(0.6 * size(val21,1));
second_split_twentyone = round(0.8 * size(val21,1));

first_split_twentytwo = round(0.6 * size(val22,1));
second_split_twentytwo = round(0.8 * size(val22,1));

first_split_twentythree = round(0.6 * size(val23,1));
second_split_twentythree = round(0.8 * size(val23,1));

first_split_twentyfour = round(0.6 * size(val24,1));
second_split_twentyfour = round(0.8 * size(val24,1));


first_split_twentyfive = round(0.6 * size(val25,1));
second_split_twentyfive = round(0.8 * size(val25,1));

first_split_twentysix = round(0.6 * size(val26,1));
second_split_twentysix = round(0.8 * size(val26,1));


% 60% for training, 20% for validation, 20% for checking
training_data = [val1(1:first_split_one, :); val2(1:first_split_two, :); val3(1:first_split_three,:) ;val4(1:first_split_four, :); val5(1:first_split_five, :); val6(1:first_split_six, :);val7(1:first_split_seven, :); val8(1:first_split_eight, :); val9(1:first_split_nine,:);val10(1:first_split_ten, :); val11(1:first_split_eleven, :); val12(1:first_split_twelve,:); val13(1:first_split_thirteen, :); val14(1:first_split_fourteen, :); val15(1:first_split_fifteen,:) ;val16(1:first_split_sixteen, :); val17(1:first_split_seventeen, :); val18(1:first_split_eighteen, :);val19(1:first_split_nineteen, :); val20(1:first_split_twenty, :); val21(1:first_split_twentyone,:);val22(1:first_split_twentytwo, :); val23(1:first_split_twentythree, :); val24(1:first_split_twentyfour,:); val25(1:first_split_twentyfive, :); val26(1:first_split_twentysix,:)];
evaluation_data = [val1(first_split_one + 1:second_split_one, :); val2(first_split_two + 1:second_split_two, :); val3(first_split_three + 1:second_split_three, :);val4(first_split_four + 1:second_split_four, :); val5(first_split_five + 1:second_split_five, :); val6(first_split_six + 1:second_split_six, :);val7(first_split_seven + 1:second_split_seven, :); val8(first_split_eight + 1:second_split_eight, :); val9(first_split_nine + 1:second_split_nine, :);val10(first_split_ten + 1:second_split_ten, :); val11(first_split_eleven + 1:second_split_eleven, :); val12(first_split_twelve + 1:second_split_twelve, :);val13(first_split_thirteen + 1:second_split_thirteen, :); val14(first_split_fourteen + 1:second_split_fourteen, :); val15(first_split_fifteen + 1:second_split_fifteen, :);val16(first_split_sixteen + 1:second_split_sixteen, :); val17(first_split_seventeen + 1:second_split_seventeen, :); val18(first_split_eighteen + 1:second_split_eighteen, :);val19(first_split_nineteen + 1:second_split_nineteen, :); val20(first_split_twenty + 1:second_split_twenty, :); val21(first_split_twentyone + 1:second_split_twentyone, :);val22(first_split_twentytwo + 1:second_split_twentytwo, :); val23(first_split_twentythree + 1:second_split_twentythree, :); val24(first_split_twentyfour + 1:second_split_twentyfour, :);val25(first_split_twentyfive + 1:second_split_twentyfive, :);val26(first_split_twentysix + 1:second_split_twentysix, :)];
testing_data = [val1(second_split_one + 1:end, :); val2(second_split_two + 1:end, :); val3(second_split_three + 1:end, :);val4(second_split_four + 1:end, :); val5(second_split_five + 1:end, :); val6(second_split_six + 1:end, :);val7(second_split_seven + 1:end, :); val8(second_split_eight + 1:end, :); val9(second_split_nine + 1:end, :);val10(second_split_ten + 1:end, :); val11(second_split_eleven + 1:end, :); val12(second_split_twelve + 1:end, :);val13(second_split_thirteen + 1:end, :); val14(second_split_fourteen + 1:end, :); val15(second_split_fifteen + 1:end, :);val16(second_split_sixteen + 1:end, :); val17(second_split_seventeen + 1:end, :); val18(second_split_eighteen + 1:end, :);val19(second_split_nineteen + 1:end, :); val20(second_split_twenty + 1:end, :); val21(second_split_twentyone + 1:end, :);val22(second_split_twentytwo + 1:end, :); val23(second_split_twentythree + 1:end, :); val24(second_split_twentyfour + 1:end, :);val25(second_split_twentyfive + 1:end, :);val26(second_split_twentysix + 1:end, :)];


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

%% Relief selection of predictors
%efarmozoume ton algorithmo tou relief gia tin epilogi xaraktiristikwn

[idx,weights] =relieff(data(:,1:end-1),data(:,end),100);
relief_array=zeros(length(idx),2);
relief_array(:,1)=idx;
 relief_array(:,2)=weights;
[relief_array,index] = sortrows(relief_array,2,'descend');

numofPred=10;


%% chosen predictors

input_training_data=training_data(:,relief_array(1:numofPred,1));
output_training_data=training_data(:,end);

input_evaluation_data=evaluation_data(:,relief_array(1:numofPred,1));
output_evaluation_data=evaluation_data(:,end);

input_testing_data=testing_data(:,relief_array(1:numofPred,1));
output_testing_data=testing_data(:,end);

%% TRAIN THE MODEL 
%FINAL MODEL 
%number of predictors : 10
%number of rules : 5

 genfis_opt = genfisOptions('FCMClustering','FISType','sugeno');
 
 genfis_opt.NumClusters=5;

initial_fis = genfis(input_training_data,output_training_data,genfis_opt);

%plot initial member functions
figure;

[x,mf] = plotmf(initial_fis,'input',1);
subplot(2,2,1);
plot(x,mf);
title('input1 ');
ylabel('Membership function');
saveas(gcf,'final_model_input1_initMF.png');

[x,mf] = plotmf(initial_fis,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 ');
ylabel('Membership function');
saveas(gcf,'final_model_input2_initMF.png');

[x,mf] = plotmf(initial_fis,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3');
ylabel('Membership function');
saveas(gcf,'final_model_input3_initMF.png');

[x,mf] = plotmf(initial_fis,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input4');
ylabel('Membership function');
saveas(gcf,'final_model_input4_initMF.png');


anfis_opt=anfisOptions('InitialFIS',initial_fis);
anfis_opt.ValidationData=[input_evaluation_data output_evaluation_data];
anfis_opt.EpochNumber=200;
[train_fis,trainError,stepSize,chkFIS,chkError] = anfis([input_training_data output_training_data],anfis_opt);


system_output = evalfis(input_testing_data,chkFIS);
system_output=round(system_output);
system_output(system_output < 1) = 1;
system_output(system_output > 26) = 26;
error = sum((system_output - output_testing_data) .^ 2);


%plotting final mfs to compare with initial

figure;
[x,mf] = plotmf(chkFIS,'input',1);
subplot(2,2,1);
plot(x,mf);
title('input1');
ylabel('Membership function');
saveas(gcf,'final_model_input1_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',2);
subplot(2,2,2);
plot(x,mf);
title('input2 ');
ylabel('Membership function');
saveas(gcf,'final_model_input2_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',3);
subplot(2,2,3);
plot(x,mf);
title('input3 ');
ylabel('Membership function');
saveas(gcf,'final_model_input3_finalMF.png');

[x,mf] = plotmf(chkFIS,'input',4);
subplot(2,2,4);
plot(x,mf);
title('input3 ');
ylabel('Membership function');
saveas(gcf,'final_model_input3_finalMF.png');


% %% MODEL EVALUATION
%error matrix
N=length(output_testing_data);
error_matrix = confusionmat(output_testing_data, system_output)

%overal accuracy
correct=zeros(1,26)
for i=1: 26
    correct(i)=error_matrix(i,i);
end

overal_accuracy=sum(correct)/N;
%user and producer accuracy

PA=zeros(1,12);
UA=zeros(1,12);
for i=1 : 12
    PA(i)=error_matrix(i,i)/length(output_testing_data(output_testing_data(:,end)==i));
    UA(i)=error_matrix(i,i)/length(system_output(system_output==i));
end

%k metric

sum1=zeros(1,12);
sum2=zeros(1,12);

for i=1:12
    correct(i)=error_matrix(i,i);
    mul1(i)= length(system_output(system_output==i))*length(output_testing_data(output_testing_data(:,end)==i));
    mul2(i)= length(output_testing_data(output_testing_data(:,end)==i))*length(system_output(system_output==i));
end

K=(N*sum(correct)-sum(mul1))/(N^2-sum(mul2));


fprintf(' OA = %f PA = %f UA = %f K = %f \n',overal_accuracy, PA, UA, K );%% PLOTTING RESULTS



%% PLOTTING RESULTS



%fuzzy system results

figure;

plot(1:length(system_output),system_output,'r*',1:length(system_output),output_testing_data(1:length(system_output),end),'b*');
title('Fuzzy System output');
legend('Fuzzy System Output','Testing data Output');
saveas(gcf,'FuzzyOutput_final_cls_model.png');

%training and evaluating errors

figure;

plot(1:length(trainError),trainError,1:length(trainError),chkError);
title('Learning Curves');
legend('Training Error', 'Testing Error');
saveas(gcf,'TrainingErrors_final_cls_model.png');


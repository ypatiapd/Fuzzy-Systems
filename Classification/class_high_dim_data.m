%% High dimensionality dataset
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
training_data = [val1(1:first_split_one, :); val2(1:first_split_two, :); val3(1:first_split_three,:) ;val4(1:first_split_four, :); val5(1:first_split_five, :); val6(1:first_split_six, :);val7(1:first_split_seven, :); val8(1:first_split_eight, :); val9(1:first_split_nine,:);val10(1:first_split_ten, :); val11(1:first_split_eleven, :); val12(1:first_split_twelve,:); val13(1:first_split_thirteen, :); val14(1:first_split_fourteen, :); val15(1:first_split_fifteen,:) ;val6(1:first_split_sixteen, :); val7(1:first_split_seventeen, :); val8(1:first_split_eighteen, :);val9(1:first_split_nineteen, :); val20(1:first_split_twenty, :); val21(1:first_split_twentyone,:);val22(1:first_split_twentytwo, :); val23(1:first_split_twentythree, :); val24(1:first_split_twentyfour,:); val25(1:first_split_twentyfive, :); val26(1:first_split_twentysix,:)];
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
%with relief algorithm we  find the most important characteristics
% we store the weigts and the indexes 

[idx,weights] =relieff(training_data(:,1:end-1),training_data(:,end),100);
relief_array=zeros(length(idx),2);
relief_array(:,1)=idx;
 relief_array(:,2)=weights;
[relief_array,index] = sortrows(relief_array,2,'descend');


%% Grid search
%number of rules and features
% NR=[3,4,5];
% NF=[3,4,5,6];
NR=[3,5,7,9,12];
NF=[3,5,7,10];
errors_array=zeros(length(NF),length(NR));%array for final errors of each rule-feature combination

for i = 1:length(NF)
    for j=1:length(NR)
        
        
     Sets = cvpartition(training_data(:, end), 'KFold', 5);%create test sets
        cvSetError = zeros(Sets.NumTestSets,1);%array of errors for sets of cvpartition

        
         genfis_opt = genfisOptions('FCMClustering','FISType','sugeno');
         genfis_opt.NumClusters = NR(j);%set the number of rules in j iteration
         initial_fis = genfis(training_data(:,relief_array(1:NF(i),1)),training_data(:,end),genfis_opt);%as input we give the features that relief algorithm selected

        %for each set of parameters we have k iterations,one for each set
        %of data partitioning
        for k=1:Sets.NumTestSets
            
            training_set=Sets.training(k);
            testing_set=Sets.test(k);
            
            training_input=training_data(training_set,relief_array(1:NF(i),1));
            training_output=training_data(training_set,end);
            
            testing_input=training_data(testing_set,relief_array(1:NF(i),1));
            testing_output=training_data(testing_set,end);
            
            anfis_opt = anfisOptions('InitialFIS', initial_fis, 'EpochNumber', 40, 'ValidationData', [testing_input testing_output]);
            
            [train_fis,trainError,stepSize,chkFIS,chkError] = anfis([training_input training_output],anfis_opt);
            
            system_output = evalfis(chkFIS,testing_data(:,relief_array(1:NF(i),1)));
            
            system_output=round(system_output);
            system_output(system_output < 1) = 1;
            system_output(system_output > 26) = 26;

            
            cvSetError(k)=sum((system_output - testing_data(:, end)).^2)/length(system_output);
            
        end
        
      model_error=sum(cvSetError)/Sets.NumTestSets;
      errors_array(i,j)=model_error(1)/length(system_output);%store the error at the final error array for all the pairs
    end 
end
%% Plot the results
%we create a bar diagram for each value of the feature parameter,and the
%values of the rules parameter
figure;
subplot(2,2,1);
bar(errors_array(1,:));
title('error with 3 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'})
saveas(gcf, 'error_3_pr.png');
      
subplot(2,2,2);
bar(errors_array(2,:));
title('error with 5 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'})
saveas(gcf, 'error_5_pr.png');

subplot(2,2,3);
bar(errors_array(3,:));
title('error with 7 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'})
saveas(gcf, 'error_7_pr.png');

subplot(2,2,4);
bar(errors_array(4,:));
title('error with 10 predictors');
xlabel('number of rules');
ylabel('error');
xticklabels({'3','5','7','9','12'});
saveas(gcf, 'error_10_pr.png');


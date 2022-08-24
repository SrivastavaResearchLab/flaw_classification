%%% Find weird signals %%%
clc
clear all
close all

data = xlsread('Training data/Type_3_529.xlsx');
data(all(isnan(data), 2), :) = [];

%% Check the signals one by one
figure
hold on
for i = 525:529
    plot(data(i,:))
    legend(num2str(i))
    pause(1);
end


%% Type 2: In total 32
weird_stuff = [9, 19, 37, 88, 149, 176, 200, 236, 247, 253, 268, 276, ...
         279, 283, 346, 347, 350, 354, 359, 363, 366, 369, 411, 414, 417,...
         430, 433, 462, 465, 490, 509, 522];
figure
hold on
for i = 530:536
    if ~(ismember(i, weird_stuff))
        plot(data(i,:))
    end
end

%% Type 3: In total 21
weird_stuff = [5, 16, 24, 38, 132, 149, 150, 172, 193, 277, 312, 330,...
    331, 384, 422, 431, 479, 487, 505, 509, 528];
figure
hold on
for i = 504:529
    if ~(ismember(i, weird_stuff))
        plot(data(i,:))
    end
end
%% Type 4: In total 1
weird_stuff = [79];
figure
hold on
for i = 410:510
    if ~(ismember(i, weird_stuff))
        plot(data(i,:))
    end
end
%% Type 5: In total 7
weird_stuff = [44, 91, 140, 148, 196, 285, 335];
figure
hold on
for i = 414:509
    if ~(ismember(i, weird_stuff))
        plot(data(i,:))
    end
end
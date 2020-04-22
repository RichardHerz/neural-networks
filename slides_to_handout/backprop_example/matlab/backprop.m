% generate data for figure to illustrate
% gradient descent method
%
% SEE backprop_createfigure.m for code
% to annotate figure 
%
% didn't use that directly but copied so code here
% have to resize figure window to get things to line up

clc 

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

xo = 0.2;
yo = 0.4;
x = linspace(-1,2,1000);
y = (x - xo).^2 + yo;

plot(x,y)
axis([0,2,0.2,2])
% dy/dx = 2*(x - xo)
x1 = 1.25
y1 = (x1 - xo).^2 + yo
% y = mx + b
% b = y - mx
m = 2*(x1 - xo) % m = dy/dx at x1,y1
b = y1 - m*x1;

hold on
plot(x1,y1,'x')
y2 = m*x + b;
plot(x,y2,'r--')

alpha = 0.2;
x3 = x1 - alpha*m
y3 = (x3 - xo).^2 + yo
plot(x3,y3,'rx')

m3 = 2*(x3 - xo)
b3 = y3 - m3*x3;
y4 = m3*x + b3;
plot(x,y4,'k--')

x5 = x3 - alpha * m3
y5 = (x5 - xo).^2 + yo
plot(x5,y5,'kx')
hold off

% Create ylabel
ylabel('ERROR');

% Create xlabel
xlabel('W');

% Create title
title({'concept of Gradient Descent method'});

% Create textbox
annotation(figure1,'textbox',...
    [0.528722157092613 0.7089173553719 0.0756154747948418 0.0933884297520661],...
    'String',{'Trial 1','W_1 = 1.25','E_1 = 1.50'});

% Create textbox
annotation(figure1,'textbox',...
    [0.246189917936691 0.390735537190082 0.199296600234467 0.100826446280992],...
    'String',{'Trial 2','W_2 = 1.25 - 0.2*(dE/dW)_{W_1} = 0.83','E_2 = 0.80'});

% Create textbox
annotation(figure1,'textbox',...
    [0.168815943728019 0.158504132231405 0.0814771395076201 0.0371900826446281],...
    'String',{'minimum E'},...
    'LineStyle','none');

% Create textbox
annotation(figure1,'textbox',...
    [0.596717467760834 0.51966115702479 0.114888628370457 0.0727272727272728],...
    'String',{'Trial 2 gradient','(dE/dW)_{W_2} = 1.26'});

% Create textbox
annotation(figure1,'textbox',...
    [0.151230949589681 0.275033057851238 0.199296600234467 0.100826446280992],...
    'String',{'Trial 3','W_3 = 0.83 - 0.2*(dE/dW)_{W_2} = 0.58','E_3 = 0.54'});

% Create textbox
annotation(figure1,'textbox',...
    [0.464243845252036 0.258504132231401 0.107854630715123 0.0727272727272728],...
    'String',{'Trial 1 gradient','(dE/dW)_{W_1} = 2.1'});

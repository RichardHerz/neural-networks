function createfigure(X1, YMatrix1, X2, Y1, X3, Y2, X4, Y3)
%CREATEFIGURE(X1, YMatrix1, X2, Y1, X3, Y2, X4, Y3)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data
%  X2:  vector of x data
%  Y1:  vector of y data
%  X3:  vector of x data
%  Y2:  vector of y data
%  X4:  vector of x data
%  Y3:  vector of y data

%  Auto-generated by MATLAB on 22-Apr-2020 15:42:12

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'Parent',axes1,'LineStyle','--');
set(plot1(1),'LineStyle','-');
set(plot1(2),'Color',[1 0 0]);
set(plot1(3),'Color',[0 0 0]);

% Create plot
plot(X2,Y1,'Marker','x','LineStyle','none');

% Create plot
plot(X3,Y2,'Marker','x','LineStyle','none','Color',[1 0 0]);

% Create plot
plot(X4,Y3,'Marker','x','LineStyle','none','Color',[0 0 0]);

% Create ylabel
ylabel('ERROR');

% Create xlabel
xlabel('W');

% Create title
title({'concept of Gradient Descent method'});

% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[0 2]);
% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes1,[0.2 2]);
box(axes1,'on');
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


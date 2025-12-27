%
% code by Richard K. Herz, github.com/RichardHerz, www.ReactorLab.net 
% 

% neural network example 
% example of a 2x2 "touchscreen" 
% the network is trained to detect vertical, horizontal & diagonal "lines" 
%
% with modifications, this algorithm is based on 
% pp. 29-32 of Chap5.3-BackProp.pdf by Sargur Srihari 
% lesson 5.3 of https://cedar.buffalo.edu/~srihari/CSE574/ 

% FOR GRADIENT DESCENT METHOD used in BACK PROPAGATION see 
% https://en.wikipedia.org/wiki/Gradient_descent 

% THIS VERSION HAS A BIAS FOR EACH INDIVIDUAL HIDDEN AND OUTPUT NODE

% >>>> THERE ARE SEVERAL CODE SECTIONS BELOW <<<<<<<

fprintf('------------ run separator ------------ \n')
clear all 
close all 
clc

numInputNodes = 4;
numOutputNodes = 4;

% numHiddenNodes = 60; % nodes per hidden layer
% numHiddenLayers = 4;

% numHiddenNodes = 20; % nodes per hidden layer
% numHiddenLayers = 3;

% there are 16 different ways the 2x2 touchscreen can be configured,
% see training inputs array below,
% so seems like 1st hidden layer should have 16 nodes
numHiddenNodes = 16; % nodes per hidden layer
numHiddenLayers = 2;

% Learning rate alpha 
alpha = 0.01;
% Lambda is for regularization 
lambda = 0.001;
% Num of iterations 
numepochs = 1e4;
% size of batches of inputs to train at same time
batchsize = 4; % e.g., 4 for 4 random examples per batch

% answers in order of train_y indices
answer = ["none" "diagonal" "vertical" "horizontal"]; 

%% Training 

% training inputs
train_x = [1 0 0 1 % diag
           0 1 1 0 % diag
           1 1 0 0 % vert
           0 0 1 1 % vert
           1 0 1 0 % horiz
           0 1 0 1 % horiz
           1 1 1 1 % all dots
           0 0 0 0 % no dots
           0 1 1 1 % start 3 dots
           1 0 1 1
           1 1 0 1
           1 1 1 0
           1 0 0 0 % start single dots 
           0 1 0 0
           0 0 1 0
           0 0 0 1];
% put examples in columns
train_x = train_x';

% number of batches
numbatches = floor( size(train_x, 2) / batchsize ); % need integer so floor
           
% training outputs as array
train_y = [0 1 0 0 % diag
           0 1 0 0 % diag
           0 0 1 0 % vert
           0 0 1 0 % vert
           0 0 0 1 % horiz
           0 0 0 1]; % horiz
% put examples in columns
train_y = train_y';
% the rest of the output y's are the columns [1 0 0 0]'
train_1 = ones(1,10);
train_0 = zeros(3,10);
train_10 = [train_1; train_0];
% combine to full train_y
train_y = [train_y train_10];

% initialize node activations
ai = zeros(numInputNodes,batchsize);
ah = zeros(numHiddenNodes,batchsize); 
ao = zeros(numOutputNodes,batchsize);
a = {ai};
for j = 2:numHiddenLayers+1
    a{j} = ah;
end
a{numHiddenLayers + 2} = ao;

% initialize connection weights randomly in range -1 to +1
wi = 2*rand(numHiddenNodes,numInputNodes) - 1;
wh = 2*rand(numHiddenNodes,numHiddenNodes) - 1;
wo = 2*rand(numOutputNodes,numHiddenNodes) - 1;

if (numHiddenLayers > 1)
    W = {wi};
    for j = 2:numHiddenLayers
        W{j} = wh;
    end
    W{numHiddenLayers + 1} = wo;
elseif (numHiddenLayers == 1)
    W{1} = wi;
    W{2} = wo;
else
    fprintf(' xxxx this program requires at least 1 hidden layer xxxxx \n')
    return
end

% intialize biases for each node 
% EITHER all zeros
% bh = zeros(numHiddenNodes,1); 
% bo = zeros(numOutputNodes,1);
% OR slightly off zero
bh = 0.01 * ones(numHiddenNodes,1); 
bo = 0.01 * ones(numOutputNodes,1);
for j = 1:numHiddenLayers
    b{j} = bh;
end
b{numHiddenLayers + 1} = bo; 

for j = 1:numHiddenLayers
    b{j} = bh;
end
b{numHiddenLayers + 1} = bo;

for j = 1 : numepochs
    % randomly rearrange the training data for each epoch
    % We keep the shuffled index in kk, so that the input and output could
    % be matched together
    kk = randperm(size(train_x, 2)); 

    for batch = 1 : numbatches 

        a{1} = train_x(:, kk( (batch-1)*batchsize+1 : batch*batchsize ) ); 
        y = train_y(:, kk( (batch-1)*batchsize+1 : batch*batchsize ) );
        
        %{
        Note: when two or more matrices are put into one cell of a
        cell array, they are combined into a single matrix in that cell;
        e.g., 4 examples of 2 input arrays in one cell creates a 2x4 array
        in that cell
        %}
               
        % Forward propagation
        
        for j = 2 : numHiddenLayers + 2
            % without biases b
            % a{j} = sigmaFunc( W{j-1}*a{j-1} );
            
            % NEW with biases b 
            a{j} = sigmaFunc( W{j-1}*a{j-1} + b{j-1});
        end
              
        %{
        Start Back-Propagation in order to train network and
        minimize error by adjusting weights using gradient
        descent method
         
        Relationships in this XOR network: 

        layer           input      hidden        output
        activation      a{1}         a{2}          a{3} to approx y  
        weight                W{1}         W{2} 

        Error
        
          j = numHiddenLayers+1
          E = 0.5 * ( y - a{j+1} ).^2
        
        Derivative of error with respect to output
        
          dE_da{j+1} = - ( y - a{numHiddenLayers+2} ) 
        
        Define inputs I from last hidden layer going to the output layer
     
          I{j} = W{j} * a{j}
        
          a{j+1} = sigmaFunc( I{j} )
          da{j+1}_dI{j} = dsigmaFunc(I{j})_dI{j}
                        = a{j+1} .* (1 - a{j+1})
        
          dE_dI{j} = dE_da{j+1} * da{j+1}_dI{j}
                   = -(y - a{j+1}) * a{j+1} .* (1 - a{j+1})
        %} 
        
        j = numHiddenLayers+1;
        dE_dI{j} = -(y - a{j+1}) .* a{j+1} .* (1 - a{j+1});

        %{
        moving "back" toward the input, 
        
        for j = numHiddenLayers : -1 : 1
        
          I{j} = W{j} * a{j}
          a{j+1} = sigmaFunc(I{j})
          da{j+1}_dI{j} = dsigmaFunc(I{j})_dI{j}
                        = a{j+1} .* (1 - a{j+1})
        
          dE_da{j+1} = dI{j+1}_da{j+1} * dE_I{j+1}
          dI{j+1}_da{j+1} = W{j+1} 
          dE_da{j+1} = W{j+1}' * dE_I{j+1}
                
          dE_dI{j} = dE_da{j+1} * da{j+1}_dI{j}
                   = W{j+1}' * dE_I{j+1} * a{j+1} .* (1 - a{j+1})
        end
        %}
        
        for j = numHiddenLayers : -1 : 1
            dE_dI{j} = W{j+1}' * dE_dI{j+1} .* a{j+1} .* (1 - a{j+1});
        end 

        %{
        Update connection weights using the gradient descent method
        
        for j = 1 : numHiddenLayers+1
        
          dE_dW{j} = dE_dI{j} * dI{j}_dW{j}
          dI{j}_dW{j} = a{j}
        
          dE_dW{j} = dE_dI{j} * a{j}' << the gradients of error w/r weights
        
          W{j} = W{j} - alpha * dE_dW{j}  << without L2 regularization
 
        end
        %}

        for j = 1 : numHiddenLayers+1
            dE_dW{j} = dE_dI{j} * a{j}';
            % L2 regularization is used for W, which is the lambda * W term 
            W{j} = W{j} - alpha * (dE_dW{j} - lambda * W{j}); 
        end

        % update biases
        for i = 1 : numHiddenLayers+1
            % get mean along rows of each node's batch element results
            %   mean(x,2) operates along DIM 2 (across the columns), 
            %   producing mean of each row of x
            %   X(DIM 1, DIM 2) >> x(row, col)
            b{i} = b{i} - alpha * mean(dE_dI{i},2);
        end
    end
end

% save workspace so we can use below after clearing 
save('WS')

%% Testing

% just loading same workspace but is useful
% for re-using the weights W
% when only running this or following sections

clear all
load('WS.mat')

% check all training examples

tsum = 0; % for count of number of errors

for tn = 1:size(train_x, 2)
    
    a{1} = train_x(:,tn);
    x = a{1};
    fprintf('input: %i %i \n       %i %i \n',x(1),x(3),x(2),x(4))
    for j = 2 : numHiddenLayers + 2
        a{j} = sigmaFunc( W{j-1}*a{j-1} );
    end
    
    yex = a{numHiddenLayers + 2};
    [tMax tIex] = max(yex);
    
    % see input above: answer = ["none" "diagonal" "vertical" "horizontal
    y = train_y(:,tn);
    [tMax tIans] = max(y);
    
    if tIex == tIans
        fprintf('answer is %s \n\n',answer(tIex))
    else
        fprintf('answer is %s >> ERROR \n\n',answer(tIex))
        tsum = tsum + 1;
    end
    
end
    
if tsum == 0
    fprintf('GOOD - no errors \n\n')
else
    fprintf('BAD - %i errors found \n\n', tsum)
end

%% set an single input to image below 

% tn 1,2 = diagonal, 3,4 = vertical, 5,6 = horizontal, others = none

tn = 2; % uses tn several places below, which single input to use

% outputs are 1 = none, 2 = diagonal, 3 = vertical, 4 = horizontal 

a{1} = train_x(:,tn);
x = a{1};
fprintf('input: %i %i \n       %i %i \n',x(1),x(3),x(2),x(4))
for j = 2 : numHiddenLayers + 2
    a{j} = sigmaFunc( W{j-1}*a{j-1} );
end

yex = a{numHiddenLayers + 2};
[tMax tIex] = max(yex);

% see input above: answer = ["none" "diagonal" "vertical" "horizontal
y = train_y(:,tn);
[tMax tIans] = max(y);

if tIex == tIans
    fprintf('answer is %s \n\n',answer(tIex))
else
    fprintf('answer is %s >> ERROR \n\n',answer(tIex))
end

% get max and min for biases
maxim = -9999;
minim = 9999;
for j = 1:numHiddenLayers+1
        if max(b{j}) > maxim
            maxim = max(b{j});
        end
        if min(b{j}) < minim
            minim = min(b{j});
        end
end
fprintf('----- biases -------- \n')
fprintf('bias min = %g, bias max = %g \n',minim,maxim)

%% imaging node activation values (a)

fprintf('----- image node values -------- \n')
% get min and max of entire set so can image
% on same color scale
% for node values with sigma function, range is 0-1
% fprintf('activation min = %g, activation max = %g \n',minim,maxim)
tt = ["INPUT" "layer 1" "layer 2" "layer 3" "layer 4" "OUTPUT"];
figure('Name','Node values', 'NumberTitle','off')
for j = 1:numHiddenLayers+2
    im = a{j};
    if j == 1
        imbk = im;
        im = [];
        im = [imbk(1), imbk(3); imbk(2), imbk(4)]
    end
%     subplot(2,3,j), image(64*im), title(tt(j));
    nhl = numHiddenLayers+2;
    subplot(2,nhl,j), image(64*im), title(tt(j));
end
cm = colormap(gray(64));
cm = flipud(cm); % change 0 to white, 1 to black
colormap(cm);
% label output
subplot(2,6,6)
    text(1.55,1,'NONE')
    text(1.55,2,'DIAG')
	text(1.55,3,'VERT')
	text(1.55,4,'HORIZ')

%% imaging connection weights (W)

fprintf('----- image connection weights -------- \n')
% get min and max of entire set so can image
% on same color scale
maxim = -99;
minim = 99;
for j = 2:numHiddenLayers
    tmaxim = max(max(W{j}));
    tminim = min(min(W{j}));
    if tmaxim > maxim
        maxim = tmaxim;
    end
    if tminim < minim
        minim = tminim;
    end
end
imspan = maxim-minim;
fprintf('weight min = %g, weight max = %g \n',minim,maxim)
rcm = 64;
% colormap(jet(rcm));
figure('Name','Connection weights', 'NumberTitle','off')
for j = 1:numHiddenLayers+1
    im = W{j};
    im = rcm * (im - minim)/imspan;
    im = im'; % so now row is start node, column is end node in next layer
    subplot(1,numHiddenLayers+1,j), image(im), title(sprintf('W %i ',j));
end
cm = colormap(gray(64));
cm = flipud(cm); % change 0 to white, 1 to black
colormap(cm);

if tsum == 0
    fprintf('GOOD - no errors \n\n')
else
    fprintf('BAD - %i errors found \n\n', tsum)
end
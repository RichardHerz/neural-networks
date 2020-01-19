% neural network example
% based on pp. 29=32 of Chap5.3-BackProp.pdf by Sargur Srihari 
% lesson 5.3 of https://cedar.buffalo.edu/~srihari/CSE574/ 
% with modifications 

% with 4 hidden layers of 80 neurons each, usually get all good 
% with 4 hidden layers of 60 neurons each, most of time get all good but
% check

% >>>> THERE ARE SEVERAL CODE SECTIONS BELOW <<<<<<<

fprintf('------------ run separator ------------ \n')
clear clc
clear all 

numInputNodes = 4;
numOutputNodes = 4;
numHiddenNodes = 60; % nodes per hidden layer
numHiddenLayers = 4;

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

% initialize node values
ai = zeros(numInputNodes,batchsize);
ah = zeros(numHiddenNodes,1);
ao = zeros(numOutputNodes,batchsize);
a = {ai};
for j = 2:numHiddenLayers+1
    a{j} = ah;
end
a{numHiddenLayers + 2} = ao;

% initialize weights for numHiddenLayers & numHiddenNodes 
% randomly in range -1 to +1
wi = 2*rand(numHiddenNodes,numInputNodes) - 1;
wh = 2*rand(numHiddenNodes,numHiddenNodes) - 1;
wo = 2*rand(numOutputNodes,numHiddenNodes) - 1;
W = {wi};
for j = 2:numHiddenLayers
    W{j} = wh;
end
W{numHiddenLayers + 1} = wo;

% W{1} = sort(sort(W{1},1),2);
% W{1} = sort(W{1},1);

Winit = W; % save initialized weights

% initialize biases
initBias = 0;
Bh = initBias * ones(numHiddenNodes,1);
Bo = zeros(numOutputNodes,1);
for j = 1:numHiddenLayers
    B{j} = Bh;
end
B{numHiddenLayers+1} = Bo;
dB = B;

for j = 1 : numepochs
    % randomly rearrange the training data for each epoch
    % We keep the shuffled index in kk, so that the input and output could
    % be matched together
    kk = randperm(size(train_x, 2)); 

    for b = 1 : numbatches 

        a{1} = train_x(:, kk( (b-1)*batchsize+1 : b*batchsize ) ); 
        y = train_y(:, kk( (b-1)*batchsize+1 : b*batchsize ) );
        
        % forward propagation
        for i = 2 : numHiddenLayers + 2
            % without biases B
            % a{i} = sigmaFunc( W{i-1}*a{i-1} );
            % with biases B
            a{i} = sigmaFunc(bsxfun( @plus, W{i-1}*a{i-1}, B{i-1} ) );
        end
        
        % start back-propagation 
        % calculate the error and back-propagate the error
        % in order to update the connection weights 
        
        %{
        for the last, output nodes at numHuddenLayers+2, the error is 
        the difference between the desired training output y and the        
        computed output a; then for the preceding unit at 
        numHuddenLayers+1, the error delta is the last error at 
        numHuddenLayers+2 multiplied by the derivative of the sigmaFunc 
        with respect to the output value, which are the a*(1-a) terms
        %}
        i = numHiddenLayers+1;
        d{i+1} = (y - a{numHiddenLayers+2});
        d{i} = d{i+1} .* a{i+1} .* (1 - a{i+1});

        %{
        for all preceding nodes i, numHuddenLayers down to 1, the error 
        delta is the later d at i+1, multiplied by the weights and
        then by the derivative of the sigmaFunc with respect to the
        output value, which are the a*(1-a) terms
        %}
        for i = numHiddenLayers : -1 : 1
            d{i} = W{i+1}' * d{i+1} .* a{i+1} .* (1 - a{i+1});
        end 

        % update weights after all error delta d's have been computed
        % L2 regularization is used for W, which is the lambda * W term 
        for i = 1 : numHiddenLayers+1
            dW{i} = d{i} * a{i}';
            W{i} = W{i} + alpha * (dW{i} - lambda * W{i}); 
        end
        
%         % update biases added to nodes in hidden layers
%         for i = 1 : numHiddenLayers
%             dB{i} = sum(d{i},2);
%             B{i} = B{i} + alpha * dB{i};
%         end
        
    end
end

% save workspace so we can use below after clearing 
save('WS')

%% Testing

% just loading same workspace but could be useful
% in future for re-using the weights W
clear all
load('WS.mat')

clc

% check all training examples

tsum = 0; % for count of number of errors

for tn = 1:size(train_x, 2)
    
    a{1} = train_x(:,tn);
    x = a{1};
    fprintf('input: %i %i \n       %i %i \n',x(1),x(3),x(2),x(4))
    for i = 2 : numHiddenLayers + 2
        a{i} = sigmaFunc( W{i-1}*a{i-1} );
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

tn = 2; % uses tn several places below 

% outputs are 1 = none, 2 = diagonal, 3 = vertical, 4 = horizontal 

a{1} = train_x(:,tn);
x = a{1};
fprintf('input: %i %i \n       %i %i \n',x(1),x(3),x(2),x(4))
for i = 2 : numHiddenLayers + 2
    a{i} = sigmaFunc( W{i-1}*a{i-1} );
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
t_min = minim
t_max = maxim
imspan = maxim-minim;
% fprintf('weight min = %g, weight max = %g \n',minim,maxim)
rcm = 64;
% colormap(jet(rcm));
figure('Name','Connection weights', 'NumberTitle','off')
for j = 1:numHiddenLayers+1
    im = W{j};
    im = rcm * (im - minim)/imspan;
    im = im'; % so now row is start node, column is end node in next layer
    subplot(1,numHiddenLayers+1,j), imagesc(im), title(sprintf('W %i ',j));
end
cm = colormap(gray(64));
cm = flipud(cm); % change 0 to white, 1 to black
colormap(cm);

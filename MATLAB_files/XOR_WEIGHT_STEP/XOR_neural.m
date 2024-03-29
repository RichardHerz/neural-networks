%
% code by Richard K. Herz, github.com/RichardHerz, www.ReactorLab.net 
%

% neural network example
% example of XOR from  https://www.mladdict.com/neural-network-simulator

% >>> SPECIAL - plot weights during training - search zzz 

%}

% FOR GRADIENT DESCENT METHOD used in BACK PROPAGATION see 
% https://en.wikipedia.org/wiki/Gradient_descent 

% >>>> THERE ARE SEVERAL CODE SECTIONS BELOW <<<<<<<

fprintf('------------ run separator ------------ \n')
clear clc
clear all 

epochCounter = 0; % zzz 
epochCounterMax = 10000; % zzz interval of plotting weights
epochPlotCounter = 0; % zzz 

numInputNodes = 2;
numOutputNodes = 1;
numHiddenNodes = 3; % nodes per hidden layer, must be >= 1
numHiddenLayers = 1;

% Learning rate alpha 
alpha = 0.01;
% Lambda is for regularization 
lambda = 0.001;
% Num of iterations 
numepochs = 1e5;
% size of batches of inputs to train at same time
batchsize = 4; % e.g., 4 for 4 random examples per batch

%% Training 

% training inputs
train_x = [1 1
           0 1
           1 0
           0 0];
% put examples in columns
train_x = train_x';

% number of batches
numbatches = floor( size(train_x, 2) / batchsize ); % need integer so floor
           
% training outputs as array
train_y = [0
           1
           1
           0];
% put examples in columns
train_y = train_y';

% initialize node activations
ai = zeros(numInputNodes,batchsize);
ah = zeros(numHiddenNodes,batchsize); 
ao = zeros(numOutputNodes,batchsize);
a = {ai};
for j = 2:numHiddenLayers+1
    a{j} = ah;
end
a{numHiddenLayers + 2} = ao;

% % initialize connection weights randomly in range -1 to +1
% wi = 2*rand(numHiddenNodes,numInputNodes) - 1;
% wh = 2*rand(numHiddenNodes,numHiddenNodes) - 1;
% wo = 2*rand(numOutputNodes,numHiddenNodes) - 1;

% SPECIAL - use initial weights used in slides handout - zzz

wi = [-0.174631930303051  -0.230331066278993
   0.939106662693414   0.511878531860625
   0.232138796450696  -0.019806025322840];
wo = [-0.276847416651616   0.188384539901641  -0.582809141323093];

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
    
% >>> SPECIAL - IMAGE CONNECTION WEIGHTS - zzz

epochCounter = epochCounter + 1;
if (epochCounter ~= 1) % zzz 
    if (epochCounter >= epochCounterMax) 
        epochCounter = 0;
    end
else
    figure(1)
    
    epochPlotCounter = epochPlotCounter + 1;
    fprintf('epochPlotCounter = %i \n', epochPlotCounter-1)
    
    % report current weights
    wi = W{1}
    wo = W{2}
    
   % compute outputs and error with current weights
   % for this input aa{1}
     aa{1} = [1; 0];
        for j = 2 : numHiddenLayers + 2
            % without biases B
            % a{j} = sigmaFunc( W{j-1}*a{j-1} );
            % with biases B
            aa{j} = sigmaFunc(bsxfun( @plus, W{j-1}*aa{j-1}, B{j-1} ) );
        end
    % note for this example with only 1 output node
    % don't need sum - but leave in case copy to other 
    % networks 
    E = sum(0.5*(train_y(3) - aa{3}).^2)
    
    % USE FINAL min & main
    maxim = 15.5;
    minim = -11.2;
    % MUST USE IMAGE NOT IMAGESC 
    imspan = maxim-minim;
    rcm = 64;
    for j = 1:numHiddenLayers+1
        im = W{j};
        im = rcm * (im - minim)/imspan;
        im = im'; % so now row is start node, column is end node in next layer
        subplot(1,numHiddenLayers+1,j), image(im);
%         subplot(1,numHiddenLayers+1,j), image(im), title(sprintf('W %i ',j));
    end
    cm = colormap(gray(64));
    cm = flipud(cm); % change 0 to white, 1 to black
    colormap(cm);
%     pause(1)
    % wait so can take screenshot
    z = input('enter 1 to quit, or RETURN to continue')
    if (z == 1), return, end;
    
end % end of IF epochCounter == 1

    for b = 1 : numbatches 

        a{1} = train_x(:, kk( (b-1)*batchsize+1 : b*batchsize ) ); 
        y = train_y(:, kk( (b-1)*batchsize+1 : b*batchsize ) );
        
        %{
        Note: when two or more matrices are put into one cell of a
        cell array, they are combined into a single matrix in that cell;
        e.g., 4 examples of 2 input arrays in one cell creates a 2x4 array
        in that cell
        %}
               
        % Forward propagation
        
        for j = 2 : numHiddenLayers + 2
            % without biases B
            % a{j} = sigmaFunc( W{j-1}*a{j-1} );
            % with biases B
            a{j} = sigmaFunc(bsxfun( @plus, W{j-1}*a{j-1}, B{j-1} ) );
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

% just loading same workspace but is useful
% for re-using the weights W
% when only running this or following sections
clear all
load('WS.mat')

clc

% check all training examples

tsum = 0; % for count of number of errors

for tn = 1:size(train_x, 2)
    
    a{1} = train_x(:,tn);
    x = a{1};
    fprintf('input: %i %i \n',x(1),x(2))
    for j = 2 : numHiddenLayers + 2
        a{j} = sigmaFunc( W{j-1}*a{j-1} );
    end
    
    yex = a{numHiddenLayers + 2};
    [tMax tIex] = max(yex);
    
    y = train_y(:,tn);
    [tMax tIans] = max(y);
    
    if round(yex) == round(y)
        fprintf('answer is good %i \n',yex)
    else
        fprintf('answer is BAD %i \n\n',yex)
        tsum = tsum + 1;
    end
    
end
    
if tsum == 0
    fprintf('GOOD - no errors \n')
else
    fprintf('BAD - %i errors found \n', tsum)
end

%% set an single input to image below 

tn = 2; % uses tn several places below, which single input to use 

a{1} = train_x(:,tn);
x = a{1};
fprintf('input: %i %i \n',x(1),x(2))
for j = 2 : numHiddenLayers + 2
    a{j} = sigmaFunc( W{j-1}*a{j-1} );
end

yex = a{numHiddenLayers + 2};
[tMax tIex] = max(yex);

y = train_y(:,tn);
[tMax tIans] = max(y);

fprintf('y is %g \n',y)
fprintf('yex is %g \n',yex)

if round(y) == round(yex)
    fprintf('answer is good \n')
else
    fprintf('answer is BAD \n')
    tsum = tsum + 1;
end

%% imaging node activation values (a)

fprintf('----- image node values -------- \n')
% get min and max of entire set so can image
% on same color scale
% for node values with sigma function, range is 0-1
% fprintf('activation min = %g, activation max = %g \n',minim,maxim)
tt = ["INPUT" "layer 1" "OUTPUT"];
figure('Name','Node values', 'NumberTitle','off')
for j = 1:numHiddenLayers+2
    im = a{j};
    % subplot(2,3,j), image(64*im), title(tt(j));
    nhl = numHiddenLayers+2;
    % subplot(2,nhl,j), image(64*im), title(tt(j));
    subplot(2,nhl,j), image(64*im);
end
cm = colormap(gray(64));
cm = flipud(cm); % change 0 to white, 1 to black
colormap(cm);

%% imaging connection weights (W)

fprintf('----- image connection weights -------- \n')
% get min and max of entire set so can image
% on same color scale
maxim = -99;
minim = 99;
for j = 2:numHiddenLayers+1
    tmaxim = max(max(W{j}))
    tminim = min(min(W{j}))
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

% compute outputs and error with current weights
% for this input aa{1}
aa{1} = [1; 0];
for j = 2 : numHiddenLayers + 2
    % without biases B
    aa{j} = sigmaFunc( W{j-1}*aa{j-1} );
end
% note for this example with only 1 output node
% don't need sum - but leave in case copy to other 
% networks 
E = sum(0.5*(train_y(3) - aa{3}).^2)

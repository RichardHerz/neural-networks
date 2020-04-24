% neural network example
% example of XOR from  https://www.mladdict.com/neural-network-simulator

% >>>> THiS IS SPECIAL - USES OLD CODE NOTATION 

% >>>> THIS IS SPECIAL - DOES TWO ITERATIONS WITH FIXED INITIAL CONDITION

clear all
clc

fprintf('------------ run separator ------------ \n')

numInputNodes = 2;
numOutputNodes = 1;
numHiddenNodes = 3; % nodes per hidden layer, must be >= 1
numHiddenLayers = 1;

% Learning rate alpha 
alpha = 0.5
% Lambda is for regularization 
lambda = 0.0
% Num of iterations 
numepochs = 2;
% size of batches of inputs to train at same time
batchsize = 1; % e.g., 4 for 4 random examples per batch

% training inputs
train_x = [1 0];
% put examples in columns
train_x = train_x';

% number of batches
numbatches = 1; % need integer so floor
           
% training outputs as array
train_y = [1];
% put examples in columns
train_y = train_y';

% initialize node values
ai = zeros(numInputNodes,batchsize);
ah = zeros(numHiddenNodes,batchsize);
ao = zeros(numOutputNodes,batchsize);
a = {ai};
for j = 2:numHiddenLayers+1
    a{j} = ah;
end
a{numHiddenLayers + 2} = ao;

% >>>> SPECIAL <<<<<<<

% % initialize weights for numHiddenLayers & numHiddenNodes 
% % randomly in range -1 to +1
% wi = 2*rand(numHiddenNodes,numInputNodes) - 1;
% wh = 2*rand(numHiddenNodes,numHiddenNodes) - 1;
% wo = 2*rand(numOutputNodes,numHiddenNodes) - 1;

% >>>> SPECIAL <<<<<<<

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

W1 = W{1}
W2 = W{2}

% initialize biases
initBias = 0;
Bh = initBias * ones(numHiddenNodes,1);
Bo = zeros(numOutputNodes,1);
for j = 1:numHiddenLayers
    B{j} = Bh;
end
B{numHiddenLayers+1} = Bo;
dB = B;

% d = {[0] [0]}
% dW = {[88] [99]}

% dW{2}

for j = 1 : numepochs
    % randomly rearrange the training data for each epoch
    % We keep the shuffled index in kk, so that the input and output could
    % be matched together
    kk = randperm(size(train_x, 2)); 
    
    jj = j

    for b = 1 : numbatches 

        a{1} = train_x; 
        y = train_y;
        
        %{
        Note: when more than one matrix is put into one cell of a
        cell array, they are combined into a single matrix in that cell;
        e.g., 4 examples of 2 input arrays in one cell creates a 2x4 array
        in that cell
        %}
        
        % Forward Propagation
        
        for i = 2 : numHiddenLayers + 2
            % without biases B
            % a{i} = sigmaFunc( W{i-1}*a{i-1} );
            % with biases B
            a{i} = sigmaFunc(bsxfun( @plus, W{i-1}*a{i-1}, B{i-1} ) );
        end
        
        bb = b
        a1 = a{1}
        a2 = a{2}
        a3 = a{3}
       
        % Back-Propagation in order to train network 
        
        i = numHiddenLayers+1;
        d{i} = -(y - a{i+1}) .* a{i+1} .* (1 - a{i+1});
        
        for i = numHiddenLayers : -1 : 1
            d{i} = W{i+1}' * d{i+1} .* a{i+1} .* (1 - a{i+1});
        end 

        for i = 1 : numHiddenLayers+1

            ii = i
    %         dd = d{i}
    %         ainv = a{i}'

    %         WWi = W{i}

            if i == 1
                dW1 = d{i} * a{i}'
                dW{1} = dW1
            else
    %             ii = i
    %             class(dW{i}) 
                dW2 = d{i} * a{i}'
    %             dW{2}
    %             ddWW2 = dW{2}
                dW{2} = dW2;
            end

    %         ddWW = dW{i} 

            WW_current = W{i}

            % L2 regularization is used for W, which is the lambda * W term 
            W{i} = W{i} - alpha * (dW{i} - lambda * W{i}); 

            WW_new = W{i}
        end
    end
end


fprintf('  DONE - NOW DO FINAL FORWARD PROP  \n')

        % forward propagation
        
        for i = 2 : numHiddenLayers + 2
            % without biases B
            % a{i} = sigmaFunc( W{i-1}*a{i-1} );
            % with biases B
            a{i} = sigmaFunc(bsxfun( @plus, W{i-1}*a{i-1}, B{i-1} ) );
        end

        a1 = a{1}
        a2 = a{2}
        a3 = a{3}

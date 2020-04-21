% neural network example
% example of XOR from  https://www.mladdict.com/neural-network-simulator
%
% algorithm based on pp. 29=32 of Chap5.3-BackProp.pdf by Sargur Srihari 
% lesson 5.3 of https://cedar.buffalo.edu/~srihari/CSE574/ 
% with modifications 

%{
Relationships in this XOR network: 

layer           input       hidden        output
activation      a{1}         a{2}          a{3} to approx y 
weight                W{1}         W{2} 

%}

% FOR GRADIENT DESCENT METHOD used in BACK PROPAGATION see 
% https://en.wikipedia.org/wiki/Gradient_descent 

% >>>> THERE ARE SEVERAL CODE SECTIONS BELOW <<<<<<<

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
        
        % forward propagation
        
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
       
        %{
        Start Back-Propagation in order to train network 

        The total error to be minimized is sum(0.5*(y - a).^2) at the output.

        The gradient descent algorithm is used to update the connection 
        weights between each pair of layers in order to minimize 
        the total error at the final output nodes. This is done in the
        "back" direction from the output layer back toward the input layer.

        Relationships in this XOR network: 

        layer           input      hidden        output
        activation      a{1}         a{2}          a{3} to approx y  
        weight                W{1}         W{2} 
        deltas                       d{1}          d{2}
        delta W               dW{1}        dW{2}

        equations 
          d{2} = -(y - a{3}) .* a{3} .* (1 - a{3});
          d{1} = W{2}' * d{2} .* a{2} .* (1 - a{2});
          dW{i} = d{i}*a{i}' 
          W{i} = W{i} - alpha * (dW{i} - lambda * W{i});

        The error gradients d{i} are the derivatives of the errors with 
        respect to the activation values. The dW{i} below, which are used to 
        adjust the weights, are the derivatives of the errors with respect to  
        the connection weights. 

        Starting at the output, and corresponding to the weights W{i} from 
        layer i to the last output layer i+1, the error gradients d{i} = the 
        negative of the derivatives of the total error with respect to the 
        estimated output activation, -(y - a{i+1}), times the derivatives of 
        the output activations a{i+1} with respect to the arguments of the 
        sigma function (the a*(1-a) terms)
        %}
        i = numHiddenLayers+1;
        d{i} = -(y - a{i+1}) .* a{i+1} .* (1 - a{i+1});

        d2 = d{2}

        %{
        moving "back" toward the input, 
        corresponding to the weights W{i} from layer i to i+1, the error
        gradient d{i} = the weights W{i+1} times the gradient d{i+1}, times 
        the derivatives of the activations a{i+1} with respect to the 
        arguments of the sigma function (the a*(1-a) terms)
        %}
        for i = numHiddenLayers : -1 : 1
            d{i} = W{i+1}' * d{i+1} .* a{i+1} .* (1 - a{i+1});
        end 

        d1 = d{1}

        %{
        update connection weights using the gradient descent method

        The error gradients d{i} are the derivatives of the errors with 
        respect to the activations a{i}. 

        The dW{i}, which are used to adjust the weights, are the derivatives 
        of the errors with respect to the connection weights.

        When the d{i} and a{i}' contain multiple examples (batchsize > 1),
        all the example arrays are combined into one array in each cell.
        For this operation, dW{i} = d{i} * a{i}', the result is one array
        of the size of a single example of W{i}.
        For 4 examples in 1 batch between 2 input nodes and 3 hidden nodes, 
        the size of d{i} is 3x4 and the size of a{i}' is 4x2, 
        and the result is a dW{i} (and W{i}) of size 3x2.
        For the output node in one node pair, its 4 d examples and 4 input node examples
        in the batch get individually multiplied and then summed to create 
        that one node pair's resultant dW.
        %} 

    %     dW{2}
    %     ddWW2 = dW{2}

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

% simplest touch screen detection of lines
%
% HERE USING procedural programming 

% USER INPUT = 2x2 array of 1 and 0
aIn = [1 0; 1 0] 

% initialize arrays of inputs 
aaTr = zeros(2,2,6); 
aaTr(:,:,1) = [1 0; 0 1]; % diag 1
aaTr(:,:,2) = [0 1; 1 0]; % diag 2
aaTr(:,:,3) = [1 1; 0 0]; % horiz 1
aaTr(:,:,4) = [0 0; 1 1]; % horiz 2
aaTr(:,:,5) = [1 0; 1 0]; % vert 1
aaTr(:,:,6) = [0 1; 0 1]; % vert 2 

% initialize output index to 1
onum = 1;

% initialize string array of outputs
saOut = ["none" "diagonal" "horizontal" "vertical"];

% analyze input 
if isequal(aIn, aaTr(:,:,1)) || isequal(aIn, aaTr(:,:,2))
    onum = 2;
elseif isequal(aIn, aaTr(:,:,3)) || isequal(aIn, aaTr(:,:,4))
    onum = 3;
elseif isequal(aIn, aaTr(:,:,5)) || isequal(aIn, aaTr(:,:,6))
    onum = 4;
else
    onum = 1;
end
disp("result = " + saOut(onum))


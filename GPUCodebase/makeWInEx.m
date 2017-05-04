%{
Instantiates a random WInEx (numEx x numIn) connectivity matrix
requires global numEx, numIn
returns WInEx
%}

function WInEx = makeWInEx(numEx, numIn)
WInEx = randn(numEx,numIn);
end
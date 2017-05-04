%{
Instantiates a random WExOut (numEx x numOut) connectivity matrix, scaled by sqrt(1/N)
Requires global numEx, numOut
returns WExOut
%}

function WExOut = makeWExOut(numEx, numOut)
WExOut = randn(numEx,numOut)*sqrt(1/numEx);
end
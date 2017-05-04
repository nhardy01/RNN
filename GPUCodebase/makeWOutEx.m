%{
Instantiates a random WOutEx (numOut x numEx) matrix
requires global numEx, numOut
Returns WOutEx
%}

function WOutEx = makeWOutEx(numEx, numOut)
WOutEx = ((rand(numOut,numEx)*2)-1);
end
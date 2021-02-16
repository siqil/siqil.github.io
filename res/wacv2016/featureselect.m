function [ i ] = featureselect( srcData, tgtTrain )
%FEATURESELECT select features based on discriminativity and similarity
%   Input:
%          srcData: source-domain training set (label in last column)
%          tgtTrain: target-domain training set (label in last column)
%   Output:
%          indices of the best 500 features
  

m=size(srcData,2);
srcLabel=srcData(:,m);
tgtLabel=tgtTrain(:,m);
srcPos=srcData(srcLabel==1,1:m-1);
srcNeg=srcData(srcLabel==0,1:m-1);
tgtPos=tgtTrain(tgtLabel==1,1:m-1);
tgtNeg=tgtTrain(tgtLabel==0,1:m-1);

% discrminativity
diffSrc=-fisher(srcPos,srcNeg);
diffTgt=-fisher(tgtPos,tgtNeg);
% similarity
diffPos=fisher(srcPos,tgtPos);
diffNeg=fisher(srcNeg,tgtNeg);

[c,ia,rankDiffSrc]=unique(diffSrc);
[c,ia,rankDiffTgt]=unique(diffTgt);
[c,ia,rankDiffPos]=unique(diffPos);
[c,ia,rankDiffNeg]=unique(diffNeg);
rankAll=rankDiffSrc+rankDiffTgt+rankDiffPos+rankDiffNeg;
% rankAll=rankDiffSrc+rankDiffTgt;
% rankAll=rankDiffPos+rankDiffNeg;

[c,i]=sort(rankAll);
i=i';
% smallest 500
i=i(1:500);
end

function s=fisher(d1, d2)
s=(mean(d1)-mean(d2)).^2./(var(d1)+var(d2));
end


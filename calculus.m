function gradWeights = calculus(Weights,bias,image,label)
gradWeights=zeros(10,1);
delta=10.0;

score=classBelongingProb(Weights,image,bias);
for j=1:10
  if(j != label)
    gradWeights(j)+=max(0,score(j)-score(label)+delta); 
    endif
  end
end
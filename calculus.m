function gradWeights = calculus(Weights,img,label,bias,score)
gradWeights=zeros(10,1);
delta=10.0;

score=classBelongingProb(Weights,img,bias);
for j=1:10
  if(j != label)
    gradWeights(j)+=max(0,score(j)-score(label)+delta); 
    endif
  end
end
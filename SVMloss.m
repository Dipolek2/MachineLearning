function loss = SVMloss(Weights,bias,image,label)
loss=0;
delta=10.0;

score=classBelongingProb(Weights,image,bias);
for j=1:10
  if(j != label)
    loss+=max(0,score(j)-score(label)+delta); 
    endif
  end
%regularization
  
%W=zeros(10,1);
%for i=1:10
%  for j=1:3072
%    W(i)+=pow2(Weights(i,j));
%    end
%  end
%  W./=0.5;
%  loss=loss+W;
end
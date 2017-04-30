function loss = softmax(Weights,x,bias,score)
score=classBelongingProb(Weights,x,bias);

%trick for better numeric stability
score-=max(score);
for i=1:10
  score(i)=exp(score(i));
  end
acc=sum(score);
loss=score./acc;
end
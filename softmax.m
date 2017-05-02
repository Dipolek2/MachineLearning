function probability = softmax(Weights,img,bias)
score=classBelongingProb(Weights,img,bias);

%trick for better numeric stability
%first scale scores to range <-1,1>
score/=max(score);
score=exp(score);

%divide each score by sum of scores
acc=sum(score);
probability=score./acc;
end
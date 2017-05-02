%softmax function computes how much image belongs to each given class
function score = classBelongingProb(Weights,x,bias)
score=zeros(10,1);

%multiply each pixel with it's weight
for i=1:10
  score(i)+=double(Weights(i,:))*double(x);
  end
score=score+bias;
end
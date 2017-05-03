%softmax function computes how much image belongs to each given class
function score = classBelongingProb(Weights,x,bias)
score=zeros(10,1);

%multiply each pixel with it's weight
score=double(Weights')*double(x');
score=score+bias;
end
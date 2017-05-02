function loss = softmaxVect(Weight,img,bias)

%create vector 50000 x 1
%size(x,2) = 50000, because of wrapping images and labels into structure
%size(x,1) = 1, because images are vertical vector
loss=zeros(size(img,2),10);

for i=1:size(img,2)
  loss(i,:)=softmax(Weight,img{i}.image,bias);
end
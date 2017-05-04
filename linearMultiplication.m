function result = linearMultiplication(Matrix,vector,bias)
result=zeros(size(vector(1)),size(Matrix(2)));
result=vector*Matrix;
result+=bias;
end
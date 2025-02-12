## Support Vector Machine

- for linear regression we have cost function as MSE
- for logistic regression we have cost function as log loss
- for SVM we have cost function as hyperplane

In SVM the objective is to find two extreme datapoints(+ve & -ve) which we take as Support Vectors such that the distance between them is maximum.

Dot product constraint optimization

- we take a point w which is the projection of x 
- xw = c 
- xw > c +ve 
- xw < c -ve
 we take -c as b

 hyperplane equation comes as xw + b = 0 
 - +ve hyperplane xw + b = 1
 - -ve hyperplane xw + b = -1

 Optimization function 
 - yi(wx + b) >= 1
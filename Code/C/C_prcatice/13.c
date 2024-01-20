#include <stdio.h>
#include <stdlib.h>
int main()
{
    double x[22]={2,3};
    double y[22]={1,2};
    double sum = 0;
    for(int i = 2;i<22;i++)
    {
        x[i]=x[i-1]+x[i-2];
        y[i]=y[i-1]+y[i-2];
        sum+=x[i-2]/y[i-2];
        printf("%lf\t%lf\t%lf\n",x[i-2],y[i-2],sum);
    }
    system("pause");
    return 0; 
}
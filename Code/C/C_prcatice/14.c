#include <stdio.h>
#include <stdlib.h>
int main()
{
    double sum = 0;
    for( int i =1;i<=20;i++)
    {
        double mul = 1;
        for(int j = 1;j<=i;j++)
        {
            mul*=j;
        }
        sum+=mul;
        printf("%d的阶乘为%.0lf,加和为%.0lf\n",i,mul,sum);
    }
    getchar();
    return 0;
}


